/*
 * 「通玄」难度档的 Worker：浏览器内 ONNX 推理 + PUCT 搜索。（「通玄」是难度名，模型对外叫 xuantian NN）
 * 对弈权重 models/xuantian_02.onnx = 内部训练产物 gen008B1 = 对外的 xuantian 02（冠军名按代递增）。
 * 分析权重 models/area_v1.onnx = gen008C（纯面积差头，**不参与对弈、不参与搜索**，只喂复盘右条读数）。
 *
 * 协议：与旧 AI Worker 兼容——收 { state, depth, id }，回 { type:'result', value:{moves,wallDir} }。
 * 追加：{type:'status', phase:'loading'|'ready'|'failed', info}、{type:'error', message}、
 *       {ping:true} -> {type:'pong', script, deps, loadError}（主线程用它 3 秒内判定 Worker 是否活着）。
 * 复盘：{analyze:true, states:[...], id} -> 每帧一条 {type:'analysis', id, partial:true, index, value}
 *       （value = {solved,pw,ac,over[,rc]}；面板那两条都是**本帧行棋方的静态形势判断**，不搜索：
 *        pw = 冠军网络 value 头的胜率（gen008B1 是 win 口径 ±1/0 ⇒ (v+1)/2 就是概率，见 NOTES §52），
 *        ac = 分析网络面积头的期望最终面积差（格，正=行棋方多），
 *        over=true 表示这帧已经终局（此时 rc = 真实终局面积差，不是估计），
 *        solved=+1 表示规则已判定（本帧存在一步制胜的着法，与样本无关）），
 *       全部跑完再补一条 {type:'analysis', id, done:true}；
 *       全部跑完再补一条 {type:'analysis', id, done:true}；
 *       {suggest:true, state, id} -> {type:'suggestion', id, value, meta}（meta.top = 前 3 着法，
 *       meta.pw/ac 同上一行的口径）。
 * depth 对本 Worker 无意义（NN 用时间预算），收到后忽略。
 *
 * 自适应：初始化时用一次真实搜索标定 ms/访问，据此选档位（budgetMs / simsMax），
 * 因此不同机器拿到的是"同一时间预算下该机跑得动的最多访问数"，而不是固定 sims。
 * ⚠ 这只作用于**对弈**与**当前帧首选**；复盘折线不搜索（见下方复盘分析通道的注释）。
 *
 * 任何环节失败（ORT/模型加载、推理异常、无合法着法）都只回 error，
 * 由主线程回退旧 AI Worker——本文件不 import 旧 AI，保持单向依赖。
 */
/* global importScripts, self, ort, SiegeEngine, NnAI */
'use strict';

const REV = 'v12';
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
const MODEL_URL = abs(`models/xuantian_02.onnx?rev=${REV}`);
// 面积差分析头。它**只服务复盘右条那一个读数**：不进搜索、不进 think()，
// 所以加载失败绝不能连累对弈 —— areaNet 保持 null，复盘只有右条空着（读数写"—"），左条照常。
const AREA_URL = abs(`models/area_v1.onnx?rev=${REV}`);
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

// 模型对外名（换代时和 index.html 里的 NN_NAME 一起改：Worker 是独立作用域，拿不到页面那个）
const NN_NAME = 'xuantian 02';
// 档位：按标定到的 ms/访问（含树开销）从上往下取第一个满足的档。
// 这里只报档位，不报模型名 —— 徽章上的「xuantian 02·极」由页面拼，名字只有一处定义。
const TIERS = [
    { name: '极', maxMsPerSim: 3,   budgetMs: 3600, simsMax: 1200 },   // 1200 = 训练/自博弈/arena 的统一口径
    { name: '高', maxMsPerSim: 8,   budgetMs: 2600, simsMax: 700 },
    { name: '中', maxMsPerSim: 20,  budgetMs: 2600, simsMax: 320 },
    { name: '初', maxMsPerSim: 1e9, budgetMs: 2400, simsMax: 140 }
];

let net = null;
let areaNet = null;        // 面积差分析头（可为 null：缺它只影响复盘右条，不影响对弈）
let areaError = null;
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

    // 分析头单独兜错：404 / 托管端 MIME / 内存不够都只记一条原因，绝不把对弈档一起拖成 failed。
    try {
        const abuf = await (await fetchChecked(AREA_URL)).arrayBuffer();
        if (abuf.byteLength < 1e6) throw new Error('分析模型文件异常 ' + abuf.byteLength);
        const asession = await ort.InferenceSession.create(abuf, {
            graphOptimizationLevel: 'all',
            executionProviders: ['wasm']
        });
        areaNet = NN.createOrtNet(ort, asession);
    } catch (err) {
        areaNet = null;
        areaError = (err && err.message) || String(err);
    }

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
                simsMax: plan.simsMax, threads, isolated, initMs: Date.now() - t0, mobile: MOBILE,
                area: areaNet ? 'ok' : areaError }
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
 *
 * 折线的语义 = **静态形势判断**：每帧只取"这个局面 + 本帧行棋方"的网络预报，不搜索（3.1.6 用户拍板）。
 * 为什么不搜：旧版每帧跑 400 访问，而那次搜索的产物一个都没进这两条读数 —— pw 用的是根上的一次前向
 * （nn_ai.js:411 `meta.rootV = ev.v`，且折线通道连 topK 都不传 ⇒ rootV 恒 undefined、
 * 必然走 netReadouts 的补前向分支）。等于花 400 倍的钱买 1 个不来自那 400 倍的钱的数。
 * 删掉之后整局 30–60 帧从 ~19s 掉进 1s 内，一次画完。
 *
 * 两个已知代价写进 NOTES §54，别再当故障查：
 *  1) 折线偏平。value 头是 L2 训出来的**条件均值**，最优解就是把领先与落后都往 50% 收（§50.3 那条警告）。
 *  2) 「一方走完好手之后胜率反而上一格」会长期存在：走前那一帧含"他可能走歪"的平均，
 *     走后那一帧是兑现之后的局面，台阶就是那份差价。要抹掉只能换成"最好可达"口径（本轮已否决）。
 */
const REPLAY = { budgetMs: 900, simsMax: 400, topK: 3, cPuct: 1.5 };   // 当前帧首选：这个必须留树，它答的是"棋"不是"形势"
const AREA_CELLS = 49;   // diff 口径 = 面积差/49 ⇒ 乘回来才是格数（标签满幅实测 ±47，见 §50.2）

// 一帧的网络读数：pw=胜率(0..1，行棋方视角)，ac=期望最终面积差(格，正=行棋方多)。缺哪个给 null。
async function netReadouts(state, p, rootV) {
    let v = rootV;
    if (!Number.isFinite(v)) {
        // 折线通道恒走这里（不搜索就没有 rootV）；suggest 通道带 rootV，而它本来就是同一次前向的值
        // ⇒ 同一局面在折线上和在当前帧读数上是同一个数，两套口径不会打架。
        try { v = (await net.evaluate(state, p)).v; } catch (err) { v = NaN; }
    }
    const pw = Number.isFinite(v) ? Math.min(1, Math.max(0, (v + 1) / 2)) : null;
    let ac = null;
    if (areaNet) {
        try {
            const av = await areaNet.evaluate(state, p);
            if (Number.isFinite(av.v)) ac = Math.min(1, Math.max(-1, av.v)) * AREA_CELLS;
        } catch (err) { /* 单帧评估失败不打断整局判读：这帧 ac 留空，面板写"未定"，不编数 */ }
    }
    return { pw, ac };
}

/*
 * 单帧判读：终局帧给写在局面里的真值；其余帧 = 一步制胜探测 + 两次前向。
 * 规则判定只保留这一条：存在一步制胜 ⇒ 胜率是**判定**的 100%，不是估计，前端因此不画"预报"字样。
 * 它不碰网络（对 K≈39 个着法各做一次 apply + terminalInfo），是纯引擎运算，保留的代价约几毫秒。
 * 反方向的「怎么都输」不证：那要把每个己方着法之后对手的每个应手都探一遍（O(K²)，
 * nn_ai.js 的 oppCanWinNext），而真到那种局面 value 头本来就报得极低、后面终局帧还会给真值 ——
 * 花整条折线的预算换一帧的确定性不值得。
 */
async function analyzeFrame(state) {
    const p = state.currentPlayer;
    const t = E.terminalInfo(state);
    if (t) return { solved: t.winner === 'draw' ? 0 : (t.winner === p ? 1 : -1),
                    rc: p === 0 ? t.s0 - t.s1 : t.s1 - t.s0, over: true };
    let solved = 0;
    const legal = E.legalActions(state, p);
    if (legal.length && NN.findImmediateWin(state, p, legal)) solved = 1;
    const { pw, ac } = await netReadouts(state, p);
    return { solved, pw, ac };
}

/*
 * 逐帧回流（partial:true, index:i）：现在一帧毫秒级，整局一瞬攒完，看起来就是"一次画完"。
 * 仍然保留逐条发 + 帧间 setTimeout(0)：一是保住"边算边亮"的旧行为（大棋谱不至于整块白一下），
 * 二是让出事件循环，别让整局任务把"翻一手"发的 suggest 排在后面。
 */
async function runAnalyze(states, id) {
    for (let i = 0; i < states.length; i++) {
        let value;
        try {
            value = await analyzeFrame(states[i]);
        } catch (err) {
            value = { solved: 0, failed: (err && err.message) || String(err) };
        }
        send({ type: 'analysis', id, partial: true, index: i, value });
        await new Promise(r => setTimeout(r, 0));
    }
    send({ type: 'analysis', id, done: true });
}

/*
 * 整局搜索判读（第二遍）：静态判读先秒出整条线，这一遍在后台逐帧跑一棵 400 访问的树，
 * 把读数替换成"搜索认为的期望"——
 *   胜率 = 根节点访问加权的 Q：V_root = Σ N(s,a)·Q(s,a) / Σ N(s,a)，再 (V+1)/2；
 *   面积 = 同一访问分布下，每个根候选走一步后面积头前向的加权和（面积头只出前向、不建自己的树）。
 * 只服务复盘折线/形势条，不参与对弈；终局帧跳过（真值已经在 values 里）。
 */
async function areaAfter(state, p, action) {
    try {
        const s = E.cloneState(state);
        E.applyAction(s, p, action);
        // 必须在 child 自己的行棋方视角下评：训练编码里"视角方=行棋方"（无回合平面），
        // 从根方视角评等于白送一个 tempo、读数系统性偏正。零和恒等式取负换回根方视角。
        const av = await areaNet.evaluate(s, s.currentPlayer);
        return Number.isFinite(av.v) ? -Math.min(1, Math.max(-1, av.v)) * AREA_CELLS : null;
    } catch (err) { return null; }
}

async function searchFrame(state) {
    const p = state.currentPlayer;
    const r = await NN.chooseAction(state, p, {
        net, budgetMs: REPLAY.budgetMs, simsMax: REPLAY.simsMax, cPuct: REPLAY.cPuct, rootDist: true
    });
    if (!r || !r.meta) return null;
    const m = r.meta;
    if (m.solved) {
        // 规则判定帧：胜率是判定值不是估计；面积给"被迫那一手走完后"的前向
        return { pw: m.solved > 0 ? 1 : 0,
                 ac: areaNet && m.bestAction != null ? await areaAfter(state, p, m.bestAction) : null,
                 solved: m.solved, searched: true };
    }
    const dist = m.rootDist || [];
    const tot = dist.reduce((s, x) => s + x.n, 0);
    if (!tot) return null;
    let vWin = 0, vArea = 0, hasArea = !!areaNet;
    for (const x of dist) {
        const w = x.n / tot;
        vWin += w * x.q;
        if (hasArea) {
            const a = await areaAfter(state, p, x.a);
            if (a === null) hasArea = false; else vArea += w * a;
        }
    }
    return { pw: (vWin + 1) / 2, ac: hasArea ? vArea : null, searched: true };
}

async function runSearchAnalyze(states, id) {
    for (let i = 0; i < states.length; i++) {
        let value = null;
        try {
            if (!E.terminalInfo(states[i])) value = await searchFrame(states[i]);
        } catch (err) { value = null; }
        send({ type: 'searchAnalysis', id, partial: true, index: i, value });
        await new Promise(r => setTimeout(r, 0));
    }
    send({ type: 'searchAnalysis', id, done: true });
}

async function runSuggest(state, id) {
    const player = state.currentPlayer;
    const r = await NN.chooseAction(state, player, {
        net, budgetMs: REPLAY.budgetMs, simsMax: REPLAY.simsMax, cPuct: REPLAY.cPuct, topK: REPLAY.topK
    });
    if (!r) { send({ type: 'suggestion', id, error: '无合法着法' }); return; }
    // 本手这一搜顺带出的读数：主线程立刻拿去画形势条，不用等整局判读跑到这一帧。
    // cfg 是给面板标题报口径用的（sims 上限/PUCT 常数/预算），规则判定帧也带上
    const rd = await netReadouts(state, player, r.meta && r.meta.rootV);
    r.meta = Object.assign({}, r.meta, rd, { cfg: { simsMax: REPLAY.simsMax, budgetMs: REPLAY.budgetMs, cPuct: REPLAY.cPuct } });
    send({ type: 'suggestion', id, value: { moves: r.moves, wallDir: r.wallDir }, meta: r.meta });
}

self.onmessage = function (e) {
    const { state, id, ping, warmup, analyze, searchAnalyze, suggest, states } = e.data || {};
    if (ping) {
        send({ type: 'pong', script: 'nn_worker.js', deps: DEPS.length, loadError });
        return;
    }
    if (loadError) {
        send({ type: 'status', phase: 'failed', message: loadError });
        if (state) send({ type: 'error', id, message: loadError });
        if (analyze) send({ type: 'analysis', id, error: loadError });
        if (searchAnalyze) send({ type: 'searchAnalysis', id, error: loadError });
        if (suggest) send({ type: 'suggestion', id, error: loadError });
        return;
    }
    if (initFailed) {
        const msg = NN_NAME + ' 不可用（已初始化失败）';
        send({ type: analyze ? 'analysis' : searchAnalyze ? 'searchAnalysis' : suggest ? 'suggestion' : 'error', id, message: msg, error: msg });
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
              : searchAnalyze ? () => runSearchAnalyze(states, id)
              : suggest ? () => runSuggest(state, id)
              : (warmup || !state) ? null
              : () => think(state, id);
    if (!job) return;
    initPromise.then(job).catch(err => {
        const message = (err && err.message) || String(err);
        send({ type: analyze ? 'analysis' : searchAnalyze ? 'searchAnalysis' : suggest ? 'suggestion' : 'error', id, message, error: message });
    });
};
