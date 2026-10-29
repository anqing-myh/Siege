/*
 * 「通玄」难度 Worker：浏览器内 ONNX 推理 + PUCT 搜索。
 * 对外品牌 = 通玄系列；权重文件 models/tongxuan_v1.onnx 即内部训练产物 gen007A3。
 *
 * 协议：与旧 AI Worker 兼容——收 { state, depth, id }，回 { type:'result', value:{moves,wallDir} }。
 * 追加：{type:'status', phase:'loading'|'ready'|'failed', info}、{type:'error', message}。
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

const REV = 'v1';
const MODEL_URL = `models/tongxuan_v1.onnx?rev=${REV}`;

importScripts(`ort/ort.wasm.min.js?rev=${REV}`, `siege_engine.js?rev=${REV}`, `nn_ai.js?rev=${REV}`);

const E = self.SiegeEngine;
const NN = self.NnAI;
const send = (m) => self.postMessage(m);

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
    ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
    ort.env.logLevel = 'error';

    const modelUrl = new URL(MODEL_URL, location.href).href;
    const resp = await fetch(modelUrl, { cache: 'force-cache' });
    if (!resp.ok) throw new Error('模型下载失败 HTTP ' + resp.status);
    const buf = await resp.arrayBuffer();
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

self.onmessage = function (e) {
    const { state, id } = e.data || {};
    if (initFailed) { send({ type: 'error', id, message: '通玄不可用（已初始化失败）' }); return; }
    if (!initPromise) {
        initPromise = init().catch(err => {
            initFailed = true;
            const message = (err && err.message) || String(err);
            send({ type: 'status', phase: 'failed', message });
            send({ type: 'error', message });
            throw err;
        });
    }
    if (!state) return;
    initPromise.then(() => think(state, id)).catch(err => {
        send({ type: 'error', id, message: (err && err.message) || String(err) });
    });
};
