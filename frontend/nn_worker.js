/*
 * 「通玄」难度 Worker：浏览器内 ONNX 推理 + PUCT 搜索。
 * 对外品牌 = 通玄系列；权重文件 models/tongxuan_v1.onnx 即内部训练产物 gen007A3。
 *
 * 协议：与旧 AI Worker 兼容——收 { state, depth, id }，回 { type:'result', value:{moves,wallDir} }。
 * 追加：{type:'status', phase:'loading'|'ready'|'failed', info}、{type:'error', message}、
 *       {ping:true} -> {type:'pong', script, deps, loadError}（主线程用它 3 秒内判定 Worker 是否活着）。
 * 复盘：{analyze:true, states:[...], id} -> 每帧一条 {type:'analysis', id, partial:true, index, value}
 *       （value = {solved,rn,rw,rc,tn,tw,td,over}；读数用 rn/rw/rc = 随机推演的样本数、
 *        推演得分与推演终局面积差均值（都取本帧行棋方视角），over=true 表示这帧已经终局，
 *        solved=±1 表示规则已判定；tn/tw/td 是搜索树里真走到终局的模拟，只作参考），
 *       全部跑完再补一条 {type:'analysis', id, done:true}；
 *       {suggest:true, state, id} -> {type:'suggestion', id, value, meta}（meta.top = 前 3 着法，
 *       meta.rn/rw/rc 同上一行的推演口径）。
 * depth 对本 Worker 无意义（NN 用时间预算），收到后忽略。
 *
 * 自适应：初始化时用一次真实搜索标定 ms/访问，据此选档位（budgetMs / simsMax），
 * 因此不同机器拿到的是"同一时间预算下该机跑得动的最多访问数"，而不是固定 sims。
 *
 * 任何环节失败（ORT/模型加载、推理异常、无合法着法）都只回 error，
 * 由主线程回退旧 AI Worker——本文件不 import 旧 AI，保持单向依赖。
 */
/* global importScripts, self, ort, SiegeEngine, NnAI */
'use strict';

const REV = 'v4';
// 主线程可能把本文件当源码取回、再用 blob: URL 起 Worker（siege.zeabur.com 上直接 new Worker(url) 会被拦）。
// blob 脚本的 location.href 是 blob:…，相对路径无法解析，所以约定由主线程注入 __NN_BASE = 页面目录。
const BASE = self.__NN_BASE || location.href;
const abs = (p) => new URL(p, BASE).href;
const short = (u) => String(u).replace(/^https?:\/\/[^/]+/, '');
async function fetchChecked(url) {
    const r = await fetch(url, { cache: 'force-cache' });
    if (!r.ok) throw new Error(short(url) + ' HTTP ' + r.status);
    return r;
}
async function toBlobUrl(url, type) {
    return URL.createObjectURL(new Blob([await (await fetchChecked(url)).arrayBuffer()], { type }));
}
const MODEL_URL = abs(`models/tongxuan_v1.onnx?rev=${REV}`);
const DEPS = [abs(`ort/ort.wasm.min.js?rev=${REV}`), abs(`siege_engine.js?rev=${REV}`), abs(`nn_ai.js?rev=${REV}`)];

const send = (m) => self.postMessage(m);

// importScripts 抛错（子资源 404 / Content-Type 不是 JS / 被 COEP 拦）默认会让整个 Worker 静默死掉，
// 主线程一条消息都收不到，表现就是"AI 永远思考中"。逐个包一层：哑死变成可回话、可显示的失败，
// 并且能说清到底是哪一个文件在托管端挂掉（用户看不到 console，只能靠徽章上的文件名）。
let loadError = null;
for (const dep of DEPS) {
    try {
        importScripts(dep);
    } catch (err) {
        loadError = `${dep.replace(/^https?:\/\/[^/]+/, '').split('?')[0]} 加载失败：` + ((err && err.message) || String(err));
        break;
    }
}
const E = self.SiegeEngine;
const NN = self.NnAI;
if (!loadError && (!E || !NN || typeof ort === 'undefined')) {
    loadError = '依赖未挂载：' + ['ort=' + typeof ort, 'SiegeEngine=' + typeof E, 'NnAI=' + typeof NN].join(' ');
}

const MOBILE = /Android|iPhone|iPad|iPod|Mobile/i.test((self.navigator && navigator.userAgent) || '');

// 档位：按标定到的 ms/访问（含树开销）从上往下取第一个满足的档
const TIERS = [
    { name: '通玄·极', maxMsPerSim: 3,   budgetMs: 3000, simsMax: 1500 },
    { name: '通玄·高', maxMsPerSim: 8,   budgetMs: 2600, simsMax: 700 },
    { name: '通玄·中', maxMsPerSim: 20,  budgetMs: 2600, simsMax: 320 },
    { name: '通玄·初', maxMsPerSim: 1e9, budgetMs: 2400, simsMax: 140 }
];

let net = null;
let plan = TIERS[TIERS.length - 1];
let initPromise = null;
let initFailed = false;

async function init() {
    if (!E || !NN) throw new Error('缺少 SiegeEngine / NnAI');
    send({ type: 'status', phase: 'loading' });

    const isolated = !!self.crossOriginIsolated;
    const hw = (self.navigator && navigator.hardwareConcurrency) || 1;
    const threads = isolated ? Math.max(1, Math.min(4, hw)) : 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = threads;
    // ORT 的后端模块是用 import() 取的（no-cors 请求），托管端开了 COEP 就会把它拦成 ERR_BLOCKED_BY_RESPONSE
    // （siege.zeabur.app 实测），而 fetch 同一个 URL 是 200。所以自己取源码转 blob 再交给 ORT；
    // wasm 同理直接喂字节，免得 Emscripten 拿到 blob 基地址后再按相对路径去找文件。
    const mjs = await toBlobUrl(abs('ort/ort-wasm-simd-threaded.mjs'), 'text/javascript');
    const wasm = await toBlobUrl(abs('ort/ort-wasm-simd-threaded.wasm'), 'application/wasm');
    ort.env.wasm.wasmPaths = { mjs, wasm };
    ort.env.logLevel = 'error';

    const buf = await (await fetchChecked(MODEL_URL)).arrayBuffer();
    if (buf.byteLength < 1e6) throw new Error('模型文件异常 ' + buf.byteLength);

    const t0 = Date.now();
    const session = await ort.InferenceSession.create(buf, {
        graphOptimizationLevel: 'all',
        executionProviders: ['wasm']
    });
    net = NN.createOrtNet(ort, session);

    // 标定：跑一次短搜索，用模块自己上报的 sims/ms 反推真实 ms/访问（含树与安全网开销）
    const st = E.newGame();
    st.currentPlayer = 1;
    let msPerSim = 12;
    const probe = await NN.chooseAction(st, 1, { net, budgetMs: 400, simsMax: 60, cPuct: 1.5 });
    if (probe && probe.meta && probe.meta.sims >= 5) {
        msPerSim = Math.max(0.2, probe.meta.ms / probe.meta.sims);
    }
    plan = TIERS.find(t => msPerSim <= t.maxMsPerSim) || TIERS[TIERS.length - 1];

    send({
        type: 'status', phase: 'ready',
        info: { tier: plan.name, msPerSim: +msPerSim.toFixed(2), budgetMs: plan.budgetMs,
                simsMax: plan.simsMax, threads, isolated, initMs: Date.now() - t0, mobile: MOBILE }
    });
}

async function think(state, id) {
    const player = state.currentPlayer;
    const legal = E.legalActions(state, player);
    if (!legal.length) { send({ type: 'error', id, message: '无合法着法' }); return; }
    send({ type: 'progress', value: 0.25 });
    const r = await NN.chooseAction(state, player, {
        net, budgetMs: plan.budgetMs, simsMax: plan.simsMax, cPuct: 1.5
    });
    send({ type: 'progress', value: 0.9 });
    if (!r || !r.moves || !r.wallDir) { send({ type: 'error', id, message: '搜索无结果' }); return; }
    send({ type: 'result', id, value: { moves: r.moves, wallDir: r.wallDir }, meta: r.meta });
}

// ---------- 复盘分析通道 ----------
/*
 * 与对局通道共用 worker 和 net，但结果走自己的 {type:'analysis'|'suggestion'} + id，
 * 主线程按 id 找回调，绝不碰 think() 的 nnAbort/进度条。
 * 复盘用的是固定小预算（不跟对局档位走）：翻一手就是一次短搜索，
 * 手感必须"立刻出"，这里要的是判读速度不是棋力。
 */
const REPLAY = { budgetMs: 900, simsMax: 400, topK: 3, rollouts: 300 };
// 整局折线：每帧一次搜索（和"前三选"同规模 400 访问）+ 300 局随机推演。
// 胜率的原料用推演而不是网络 value —— 后者训练目标里有 0.7 的胜负打底，全盘 |v|≤0.59 全在死区，
// 反解不出分差也反解不出胜率（NOTES §39/§42）。推演不碰网络，一帧 20–105ms，比那一搜便宜。
const ANALYZE = { budgetMs: 500, simsMax: 400, rollouts: 300 };

// 单帧判读：终局帧直接给判定值（不必搜），其余跑一次短搜索 + 一批推演
// rc = 推演均值下的期望最终面积差（正=本帧行棋方多），与 rw 同一批样本、同一个视角
async function analyzeFrame(state) {
    const p = state.currentPlayer;
    const t = E.terminalInfo(state);
    if (t) return { solved: t.winner === 'draw' ? 0 : (t.winner === p ? 1 : -1), rn: 0, rw: 0,
                    rc: p === 0 ? t.s0 - t.s1 : t.s1 - t.s0, over: true, tn: 0, tw: 0, td: 0 };
    const roll = NN.rolloutScore(state, ANALYZE.rollouts, 120);
    const r = await NN.chooseAction(state, p, { net, budgetMs: ANALYZE.budgetMs, simsMax: ANALYZE.simsMax, cPuct: 1.5 });
    const m = (r && r.meta) || {};
    return { solved: m.solved || 0, rn: roll.n, rw: roll.w, rc: roll.n ? roll.d / roll.n : 0,
             tn: m.tn || 0, tw: m.tw || 0, td: m.td || 0,
             sims: m.sims, ms: m.ms, why: m.why };
}

/*
 * 逐帧回流（partial:true, index:i），主线程画一条亮一条 —— 整局 30-60 帧，攒完要十几秒，
 * 一次性回会让用户盯着一片空白以为坏了。帧之间 setTimeout(0) 让出事件循环：
 * 否则"翻一手"发的 suggest 请求会被排在整局任务后面，界面就卡住了。
 */
async function runAnalyze(states, id) {
    for (let i = 0; i < states.length; i++) {
        let value;
        try {
            value = await analyzeFrame(states[i]);
        } catch (err) {
            value = { solved: 0, tn: 0, tw: 0, td: 0, failed: (err && err.message) || String(err) };
        }
        send({ type: 'analysis', id, partial: true, index: i, value });
        await new Promise(r => setTimeout(r, 0));
    }
    send({ type: 'analysis', id, done: true });
}

async function runSuggest(state, id) {
    const player = state.currentPlayer;
    const roll = NN.rolloutScore(state, REPLAY.rollouts, 120);
    const r = await NN.chooseAction(state, player, {
        net, budgetMs: REPLAY.budgetMs, simsMax: REPLAY.simsMax, cPuct: 1.5, topK: REPLAY.topK
    });
    if (!r) { send({ type: 'suggestion', id, error: '无合法着法' }); return; }
    // 本手这一搜顺带出的读数：主线程立刻拿去画形势条，不用等整局判读跑到这一帧
    r.meta = Object.assign({}, r.meta, { rn: roll.n, rw: roll.w, rc: roll.n ? roll.d / roll.n : 0 });
    send({ type: 'suggestion', id, value: { moves: r.moves, wallDir: r.wallDir }, meta: r.meta });
}

self.onmessage = function (e) {
    const { state, id, ping, warmup, analyze, suggest, states } = e.data || {};
    if (ping) {
        send({ type: 'pong', script: 'nn_worker.js', deps: DEPS.length, loadError });
        return;
    }
    if (loadError) {
        send({ type: 'status', phase: 'failed', message: loadError });
        if (state) send({ type: 'error', id, message: loadError });
        if (analyze) send({ type: 'analysis', id, error: loadError });
        if (suggest) send({ type: 'suggestion', id, error: loadError });
        return;
    }
    if (initFailed) {
        const msg = '通玄不可用（已初始化失败）';
        send({ type: analyze ? 'analysis' : suggest ? 'suggestion' : 'error', id, message: msg, error: msg });
        return;
    }
    if (!initPromise) {
        initPromise = init().catch(err => {
            initFailed = true;
            const message = (err && err.message) || String(err);
            send({ type: 'status', phase: 'failed', message });
            throw err;
        });
    }
    const job = analyze ? () => runAnalyze(states, id)
              : suggest ? () => runSuggest(state, id)
              : (warmup || !state) ? null
              : () => think(state, id);
    if (!job) return;
    initPromise.then(job).catch(err => {
        const message = (err && err.message) || String(err);
        send({ type: analyze ? 'analysis' : suggest ? 'suggestion' : 'error', id, message, error: message });
    });
};
