/*
 * NN 难度档的 Worker：浏览器内 ONNX 推理 + PUCT 搜索。（「通玄」「大师」都是难度名，模型另有对外名）
 * 3.2.3 换代（NOTES §84/§85/§86.1）：gen014_a1 过晋升门（对现役冠军 qv24 400 局 58.0% CI[53.8,64.0]
 * p=8.2e-4，另在两条独立尺上 a1 vs 前冠军 B1 59.0%、a1 vs ai4(open6) 88.5% ⇒ 不是只克现冠军），
 * 挂名 **xuantian 04**；原冠军 gen013_qv24 退档**不上任何难度**（配方逐字复刻它，只是数据换成 gen014）。
 * 对弈权重按难度分两份，主线程每条对局消息带 play:'master'|'tongxuan' 指定用哪一份：
 *   通玄 models/xuantian_04.onnx = 内部训练产物 gen014_a1（现役冠军）。
 *   大师 models/xuantian_02.onnx = 内部训练产物 gen008B1（上一代冠军，xuantian 02）。
 *   两份都**用到才下**（只点大师不会下那份 19 MB 依赖，只点通玄不会下那 4.5 MB），架构相同 ⇒ 标定一次两边共用。
 * 分析头**并入冠军**：gen014_a1 既是通玄的对弈权重，也在池尺上量到 R²=0.616 / MAE=0.296 / 斜率 1.01
 * （同一把尺上上一代冠军 qv24 是 0.604 / 0.299 / 1.00，现役大师 B1 只有 0.510 / 0.352 / 0.84）
 * ⇒ 复盘读数直接读冠军那一份，**不再单独下一份权重**。
 *   models/analysis_v1.onnx（gen013_qv16）、models/qv_v1.onnx（gen013_qv，只上线过一天的大师权重）
 *   与 models/xuantian_03.onnx（gen013_qv24，3.1.19~3.2.3 的通玄）就此**下线但留在仓库里当回滚点**，
 *   没有任何代码引用 ⇒ 浏览器不会再下载它们。
 * 复盘右条的面积差读数是**两特征线性校准的预报**（3.1.21，NOTES §73）：规则 BFS 归属差 + 冠军 value 头
 * → 期望最终面积差（格），零新增权重、零额外前向；面积分析头 models/area_v1.onnx（gen008C）下线留档。
 * 左条（胜率）与右条**共用同一次 value 前向**，所以两条天然同口径，但也天然冗余（见 §72 三）。
 *
 * 协议：与旧 AI Worker 兼容——收 { state, depth, id, play }，回 { type:'result', value:{moves,wallDir} }。
 *       play 缺省按 'tongxuan' 处理（旧调用点不必一起改）。
 * 追加：{type:'status', phase:'loading'|'ready'|'failed', info}、{type:'error', message}、
 *       {ping:true} -> {type:'pong', script, deps, loadError}（主线程用它 3 秒内判定 Worker 是否活着）。
 * 复盘：{analyze:true, states:[...], id} -> 每帧一条 {type:'analysis', id, partial:true, index, value}
 *       （value = {solved,pw,ac,over[,rc]}；面板那两条都是**本帧行棋方的静态形势判断**，不搜索：
 *        pw = 冠军网络 value 头的胜率（qv24 是 win 口径 ±1/0 ⇒ (v+1)/2 就是概率，见 NOTES §52；
 *             校准斜率实测 1.00，所以这个数不需要打任何折扣，见下方 ANALYSIS 一节），
 *        ac = **期望最终面积差**（格，正=行棋方多）= A0 + A1·规则BFS归属差 + A2·value，见 AREA_CAL；
 *             拿不到权重时退化成 A1-only 那条（B0 + B1·规则），归属差本身与训练编码 plane 9 同一口径，
 *        over=true 表示这帧已经终局（此时 rc = 真实终局面积差，不是估计），
 *        solved=+1 表示规则已判定（本帧存在一步制胜的着法，与样本无关）），
 *       全部跑完再补一条 {type:'analysis', id, done:true}；
 *       {suggest:true, state, id} -> {type:'suggestion', id, value, meta}（meta.top = 前 3 着法，
 *       meta.pw/ac 同上一行的口径。首选与读数**恒出在冠军那份对弈权重上**，与你这局玩的是哪档无关，
 *       所以复盘时大师档用户可能第一次触发冠军权重那份下载）。
 * depth 对本 Worker 无意义（NN 用时间预算），收到后忽略。
 *
 * 自适应：初始化时用一次真实搜索标定 ms/访问，据此选档位，因此不同机器拿到的是
 * "3 秒硬限内该机跑得动的最多访问数"，而不是固定 sims。3.1.20 起四档 = 1200 / 800 / 400 / 150 访问，
 * budgetMs 一律 3000（搜索循环自己卡 80% ⇒ 最坏 2.4 s 返回，加上传递与落子动画也仍在 3 s 内）；
 * 选档判据也从"固定 ms/访问阈值"改成「实测 ms/访问 × 该档 sims ≤ 2400 ms」，同一台机器同一个答案，
 * 不必再维护四个魔数。
 * ⚠ 这只作用于**对弈**与**当前帧首选**；复盘折线不搜索（见下方复盘分析通道的注释）。
 *
 * 对弈温度（3.2.1 定三段，3.2.4 把开局段 0.40→0.30）：按手数分三段 0.30 / 0.15 / 0.05，再按档位缩放，
 * 只影响**实际落子**（根的访问数分布上采样），不影响复盘的首选/前三选/任何读数 —— 那条路不传 temp。
 * 详见下方 playTemp()。arena/Elo 那把尺仍是 temp=0 量的，两个口径别混。
 *
 * 任何环节失败（ORT/模型加载、推理异常、无合法着法）都只回 error，
 * 由主线程回退旧 AI Worker——本文件不 import 旧 AI，保持单向依赖。
 */
/* global importScripts, self, ort, SiegeEngine, NnAI */
'use strict';

const REV = 'v21';
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
// 对弈权重两份，按难度取（见文件头）：通玄=冠军 xuantian 04，大师=上一代冠军 xuantian 02。都惰性加载，
// 只在下第一手/warmup 时按当条消息的 play 取那一份 ⇒ 选大师不会连带下那 19 MB 依赖。
const MODEL_URL = abs(`models/xuantian_04.onnx?rev=${REV}`);
const MASTER_URL = abs(`models/xuantian_02.onnx?rev=${REV}`);
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

// 模型对外名（换代时和 index.html 里的同名常量一起改：Worker 是独立作用域，拿不到页面那两个）
const NN_NAME = 'xuantian 04';       // 通玄（现役冠军 gen014_a1）
const NN_MASTER_NAME = 'xuantian 02';  // 大师（上一代冠军 gen008B1，退档不退名）
// play 键 -> 那一份对弈权重。name 只用于报错时点名（徽章上的名字由页面拼，一处定义）。
const PLAY = {
    master: { url: MASTER_URL, name: NN_MASTER_NAME },
    tongxuan: { url: MODEL_URL, name: NN_NAME }
};
// 档位：按标定到的 ms/访问（含树开销）从上往下取第一个满足的档。
// 这里只报档位，不报模型名 —— 徽章上的「xuantian 04·极」由页面拼，名字只有一处定义。
// 四档访问数（3.1.20）：顶档 1200 = 训练/自博弈/arena 的统一口径，别往上加 ⇒
// 1500/2000 那一段实测是"800 以上一条测不出差异的浅坡"（NOTES：n=50 阶梯 1200→1500 = 0±7.5pp、
// 1200→2000 = +4pp±8pp），买不到棋力，只会吃掉 3 秒的余量、把边缘机器整档挤下去。
// budgetMs 全部 3000：nn_ai.js 的搜索循环卡 budget*0.8 ⇒ 单手最坏 2.4 s 一定返回。
const TIERS = [
    { name: '极', simsMax: 1200, budgetMs: 3000 },
    { name: '高', simsMax: 800, budgetMs: 3000 },
    { name: '中', simsMax: 400, budgetMs: 3000 },
    { name: '初', simsMax: 150, budgetMs: 3000 }
];
const TIER_FIT_MS = 2400;      // "跑得完"的判据：实测 ms/访问 × 该档 sims ≤ 搜索自己的截止线（3000×0.8）

// 对弈网络按难度各存一份（'master'/'tongxuan'，用到才下）。复盘一律 ensurePlay('tongxuan')：
// 首选与读数只认冠军，不跟着难度走。
const nets = {};
let plan = TIERS[TIERS.length - 1];
let calibrated = 0;         // 已标定出的 ms/访问；两份对弈网络同架构 ⇒ 标一次两边共用
let initPromise = null;
let initFailed = false;

// 一份 ORT 配置两处用（两档对弈网络各按需加载）：环境（wasmPaths/线程/日志级别）在 initOrt 里全局设好。
async function loadNet(url, label) {
    const buf = await (await fetchChecked(url)).arrayBuffer();
    if (buf.byteLength < 1e6) throw new Error(label + '文件异常 ' + buf.byteLength);
    const session = await ort.InferenceSession.create(buf, {
        graphOptimizationLevel: 'all',
        executionProviders: ['wasm']
    });
    return NN.createOrtNet(ort, session);
}

/*
 * 复盘读数的胜率 = 冠军网络（xuantian 04 = gen014_a1）的 value 头，与对弈用的是同一份权重。
 * 实测在同一批 800 组、同一个终局目标 zWin 上量的（engine/ruler_head.js，NOTES §63 一/§67/§86.2）：
 *   gen014_a1      R²=0.616  MAE=0.296  斜率 1.01   ← 现役（对弈 + 读数）
 *   gen013_qv24    R²=0.604  MAE=0.299  斜率 1.00   ← 上一代冠军（3.2.3 起下线留档）
 *   gen013_qv16    R²=0.597  MAE=0.304  斜率 1.00   ← 更早的分析头，已下线留档
 *   gen008B1       R²=0.510  MAE=0.352  斜率 0.84   ← 现在的「大师」
 * 斜率才是关键：0.84 = 过自信，把领先与落后的差距整体夸大 ~19%；1.01 意味着 (v+1)/2 直接就是概率，
 * 前端一行折扣都不用打。3.1.19 之前"对弈读 A 头、复盘读 B 头"是因为冠军的读数头差，
 * 现在冠军自己就是读数最准的那份 ⇒ 两份权重并成一份，复盘不再多下一次模型。
 */

/*
 * 取某一档的对弈权重（用到才下）。这里不记住失败：权重下不来这一手就废了，
 * 让本手重试比"一次会话内静默降级"好。
 */
async function ensurePlay(which) {
    const key = PLAY[which] ? which : 'tongxuan';
    if (nets[key]) return nets[key];
    const n = await loadNet(PLAY[key].url, PLAY[key].name);
    nets[key] = n;
    return n;
}

async function initOrt() {
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
    return { threads, isolated };
}

// 标定：跑一次短搜索，用模块自己上报的 sims/ms 反推真实 ms/访问（含树与安全网开销）。
// 两档权重是同一个网络结构 ⇒ 推理开销同一量级，所以只标第一次加载的那份，之后复用。
async function calibrate(n) {
    if (calibrated) return calibrated;
    const st = E.newGame();
    st.currentPlayer = 1;
    let msPerSim = 12;
    const probe = await NN.chooseAction(st, 1, { net: n, budgetMs: 400, simsMax: 60, cPuct: 1.5 });
    if (probe && probe.meta && probe.meta.sims >= 5) {
        msPerSim = Math.max(0.2, probe.meta.ms / probe.meta.sims);
    }
    plan = TIERS.find(t => msPerSim * t.simsMax <= TIER_FIT_MS) || TIERS[TIERS.length - 1];
    calibrated = msPerSim;
    return msPerSim;
}

async function init(which) {
    if (!E || !NN) throw new Error('缺少 SiegeEngine / NnAI');
    send({ type: 'status', phase: 'loading' });

    const t0 = Date.now();
    const env = await initOrt();
    const msPerSim = await calibrate(await ensurePlay(which));

    send({
        type: 'status', phase: 'ready',
        info: { tier: plan.name, msPerSim: +msPerSim.toFixed(2), budgetMs: plan.budgetMs,
                simsMax: plan.simsMax, threads: env.threads, isolated: env.isolated,
                initMs: Date.now() - t0, mobile: MOBILE }
    });
}

/*
 * 对弈温度（3.2.1 定三段 0.40/0.15/0.05；**3.2.4 开局段收到 0.30**，3.1.20 那版是 0.50/0.25/0.05）。
 * 只作用于**实际落子**的采样，不改搜索、不改任何读数。
 *   ply < 12        τ=0.30  开局布局近似等价，留一点多样性即可；训练数据这一段用的是 τ_open=1.0 / OPEN_PLIES=12
 *                          （engine/gen_selfplay.js 口径），⇒ 0.30 比生成时的对手更稳，不至于昏招。
 *   12 ≤ ply < 24   τ=0.15  中盘：走歪一手是亏，不是输。
 *   ply ≥ 24        τ=0.05  后半盘：面积已经开始定型，一步错就是翻脸 ⇒ 基本回到 argmax。
 * 24 这个界（3.2.1 从 36 提前）＝ 双方各下满 12 手之后；实测对局中位 41 手（qv24/qv16 vs B1 共 800 局：
 * P10 28、P75 46、P90 51、max 70）⇒ 大部分对局的后一半都落在 τ=0.05 这一段。
 * 顺带：短于 28 手就打完的那些局是靠"一步制胜 / 全灭保地盘"两条规则判定收尾的，而那两条压根不经过采样。
 * 档位系数 1 / 0.8 / 0.6 / 0.4（3.2.1 定，取代原来的 min(1, simsMax/1200)）：慢档的访问分布本身就抖，
 * 同一个 τ 在抖的分布上会被放大 ⇒ 弱机器上温度自动收着。实测 184 组真访问分布（NOTES §82）：
 * 开局偏离 极 84.9% / 高 77.4% / 中 48.3% / 初 28.6%，ΔQ 全表在 ±0.006 内（走歪≠走坏）。
 * 旧式按比例缩放还有个坑：初档后半盘 τ=0.006 时 Math.pow(150,166.7) 溢出成 Infinity，
 * 抽取循环拿 NaN 比大小永不命中 ⇒ 14% 的局面静默退化成 argmax。现在最低 τ=0.02，不再溢出。
 */
function playTemp(state) {
    // turnCount 来自消息里的状态快照（主线程 cloneState 出来的）；缺字段就当开局，宁可多给一点多样性
    const tc = state.turnCount > 0 ? state.turnCount : 1;
    const ply = (tc - 1) * 2 + (state.currentPlayer === 0 ? 0 : 1);
    const base = ply < 12 ? 0.30 : ply < 24 ? 0.15 : 0.05;
    const f = plan.simsMax >= 1200 ? 1 : plan.simsMax >= 800 ? 0.8 : plan.simsMax >= 400 ? 0.6 : 0.4;
    return +(base * f).toFixed(3);
}

async function think(state, id, play) {
    const player = state.currentPlayer;
    const legal = E.legalActions(state, player);
    if (!legal.length) { send({ type: 'error', id, message: '无合法着法' }); return; }
    send({ type: 'progress', value: 0.25 });
    // 取不到权重就抛给外层 catch ⇒ 主线程照旧走"回退规则 AI + 徽章说明"，不在这里编一个弱手。
    const n = await ensurePlay(play);
    const temp = playTemp(state);
    const r = await NN.chooseAction(state, player, {
        net: n, budgetMs: plan.budgetMs, simsMax: plan.simsMax, cPuct: 1.5, temp: temp
    });
    send({ type: 'progress', value: 0.9 });
    if (!r || !r.moves || !r.wallDir) { send({ type: 'error', id, message: '搜索无结果' }); return; }
    send({ type: 'result', id, value: { moves: r.moves, wallDir: r.wallDir }, meta: r.meta });
}

// ---------- 复盘分析通道 ----------
/*
 * 与对局通道共用 worker，但复盘读数**恒取冠军那份权重**（与所选难度无关，见 netReadouts）；
 * 结果走自己的 {type:'analysis'|'suggestion'} + id，
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
 *     换到分析头（3.1.17）不消除它：那一步只把**斜率**从 0.84 修回 1.00（差距不再被夸大 ~19%）、
 *     R² 从 0.51 提到 0.60；条件均值的收缩本身是目标函数决定的，只能换口径（已否决）不能换网络。
 *  2) 「一方走完好手之后胜率反而上一格」会长期存在：走前那一帧含"他可能走歪"的平均，
 *     走后那一帧是兑现之后的局面，台阶就是那份差价。要抹掉只能换成"最好可达"口径（本轮已否决）。
 */
const REPLAY = { budgetMs: 900, simsMax: 400, topK: 3, cPuct: 1.5 };   // 当前帧首选：这个必须留树，它答的是"棋"不是"形势"

/*
 * 规则面积差：口径与判定都在 siege_engine 的 areaRaceDiff 里
 * （每格判给「BFS 步数更少」的一方，取差；单位=格，正=视角方多；与训练编码 plane 9 同一套邻接）。
 * 3.1.15 起它曾**直接**当面积条读数；3.1.21 起它只是下面那条校准的**一个自变量**（系数 A1）。
 * 单用它不够：这条免费规则在离策略池上的组内 R² 只有 0.080(ply12) / 0.102(ply24)（交叉验证 0.064 / 0.093），
 * 而专门为面积训的那只头（gen008C，跑满 20 轮）也只有 +0.064 ⇒ 面积信息的大头不在这条规则里，在 value 头里。
 */
// 薄封装（不写成 `const ruleAreaDiff = E.areaRaceDiff`）：依赖没挂载时 E 是 undefined，
// 模块作用域上直接取属性会让整个 Worker 静默死在脚本求值阶段 —— 那正是本文件从头防的那类故障。
function ruleAreaDiff(state, p) { return E.areaRaceDiff(state, p); }

/*
 * 面积条 = 两特征线性校准的**预报**（3.1.21，NOTES §73）：
 *   期望最终面积差(格) = A0 + A1·规则归属差(格) + A2·value
 * value 是冠军网络 value 头那个 ±1 的原始数（不是 (v+1)/2 的胜率）。取不到权重时退化到
 *   期望最终面积差(格) = B0 + B1·规则归属差(格)
 * 系数出处：`COEF=1 NETS=<冠军.onnx> node engine/area_ruler.js`——**这两条 pair 系数是绑着冠军 value 头
 * 拟合的，换冠军就必须重拟**（3.2.3 换成 xuantian 04 = gen014_a1 时重跑，A2 从 14.14/10.04 变到
 * 15.26/10.98；solo 那两条只吃规则，逐字未变）。拟合样本是那批 757 个离策略局面（每局面平均 20 次
 * 独立续弈 ⇒ 组内均值才是这条线的标签）；分桶 CV R²（xuantian 04 / xuantian 03 括注）：
 *   ply=12 双特征 0.141（0.129）/ 单特征 0.064；ply=24 双特征 0.392（0.313）/ 单特征 0.093。
 *   ⇒ 换代后这条线在两个桶上都更准了一点，但**单特征那条一分没动**：涨的全是 value 头的钱。
 * 为什么分两桶：**规则的权重随对局进行单调变大**。单特征回归的斜率（格/格）在离策略上是
 *   ply12=0.18 → ply24=0.31，在池内是 pt18=0.48 → pt30=0.67 → 终局前=0.85；扣掉 value 之后的偏系数 A1
 *   也跟着走（0.0743 → 0.0937）⇒ 一份全局系数不可能两头都对，18 手前后各配一档。
 * ⚠ 已知偏差：系数只在 12/24 两个点量过（离策略池就只有这两个点位），尾段（≥30 手；实测中位局长 41）
 *   规则项的真实权重大于 0.094 ⇒ 最后三分之一的条会**偏平**。要修得再下一批 pt=30/end 的续弈样本，
 *   那是生成侧的活，本轮不做（"新的数据生成先不要做"）。
 * 为什么要 +A2·value：纯规则那条免费线在 ply=24 的 CV 只有 0.093，而 value 头单独就能到 0.308
 *   ⇒ 面积信息的九成在 value 头里，规则项只是补个 tempo（详见 §72 六那张零成本菜单表）。
 */
const AREA_CAL = [
    { below: 18, pair: [-3.25, 0.0743, 15.26], solo: [-1.27, 0.1763] },
    { below: Infinity, pair: [-0.36, 0.0937, 10.98], solo: [-0.43, 0.3122] }
];

function plyOf(state, p) {
    const tc = state.turnCount > 0 ? state.turnCount : 1;
    return (tc - 1) * 2 + (p === 0 ? 0 : 1);
}

// 规则项是这条预报的主心骨（两个桶的系数都把它当必要项拟合）⇒ 它算不出来就返回 null，
// 宁可面板写"未定"也不给一个偏掉截距的数。value 缺才允许退回单特征那条。
function areaForecast(state, p, ruleCells, v) {
    if (!Number.isFinite(ruleCells)) return null;
    const bucket = AREA_CAL.find(b => plyOf(state, p) < b.below) || AREA_CAL[AREA_CAL.length - 1];
    const cells = Number.isFinite(v)
        ? bucket.pair[0] + bucket.pair[1] * ruleCells + bucket.pair[2] * v
        : bucket.solo[0] + bucket.solo[1] * ruleCells;
    return Math.max(-48, Math.min(48, cells));   // 48 = 主线程 RP_CELL_MAX（Worker 拿不到页面常量，改那边要改这里）
}

// 一帧的网络读数：pw=胜率(0..1，行棋方视角)，ac=期望最终面积差(格，正=行棋方多)。任一缺 ⇒ null。
// 取的都是**冠军**那份（不是当前难度那份）⇒ 折线口径不随难度变；只点过大师的用户此刻会把它取下来。
// rootV 的契约：只有"这一搜本身就是冠军权重跑的"才许传进来（现在唯一供者是 runSuggest）⇒ 省一次前向，
// 且同一局面在折线与当前帧上必然是同一个数。传错网络=rootV 就是别份权重的读数（探针会抓到）。
async function netReadouts(state, p, rootV) {
    let v = rootV;
    if (!Number.isFinite(v)) {
        const n = await ensurePlay('tongxuan');
        try { v = (await n.evaluate(state, p)).v; } catch (err) { v = NaN; }
    }
    const pw = Number.isFinite(v) ? Math.min(1, Math.max(0, (v + 1) / 2)) : null;
    let rule = null;
    try { rule = ruleAreaDiff(state, p); } catch (err) { /* 单帧算失败不打断整局判读：这帧 ac 留空，面板写"未定"，不编数 */ }
    return { pw, ac: areaForecast(state, p, rule, v) };
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
 * 把**胜率**替换成"搜索认为的期望"：V_root = Σ N(s,a)·Q(s,a) / Σ N(s,a)，再 (V+1)/2。
 * 面积条**不被这一遍覆盖**（3.1.21）：它的口径是"A0+A1·规则归属 + A2·静态 value"那个线性式，
 * 系数就是在静态 value 上拟合的 ⇒ 往里灌 V_root 等于拿 A 的尺量 B 的数。实测搜索对面积的增益也只有
 * CV R² +0.02 这个量级（0.312→0.329，§72 六），换不到"两条各自多跑一遍"的钱。
 * 唯一保留的例外是规则判定帧（solved）：那时胜负已定，"期望最终面积差"就是被迫那一手走完后的
 * 规则归属差，是个近乎真值的数，不是估计。
 * 只服务复盘折线/形势条，不参与对弈；终局帧跳过（真值已经在 values 里）。
 */
async function areaAfter(state, p, action) {
    try {
        const s = E.cloneState(state);
        E.applyAction(s, p, action);
        // 必须在 child 自己的行棋方视角下算：根方刚走完这一手，child 的行棋方是对手，
        // 从根方视角算等于白送一个 tempo、读数系统性偏正。零和恒等式取负换回根方视角。
        return -ruleAreaDiff(s, s.currentPlayer);
    } catch (err) { return null; }
}

async function searchFrame(state) {
    const p = state.currentPlayer;
    // 与折线同一份权重（冠军）：这一遍的产物会**覆盖**静态读数画在同一条线上，混两个口径等于造假。
    const n = await ensurePlay('tongxuan');
    const r = await NN.chooseAction(state, p, {
        net: n, budgetMs: REPLAY.budgetMs, simsMax: REPLAY.simsMax, cPuct: REPLAY.cPuct, rootDist: true
    });
    if (!r || !r.meta) return null;
    const m = r.meta;
    if (m.solved) {
        // 规则判定帧：胜率是判定值不是估计；面积给"被迫那一手走完后"的规则归属差（见上：这条是近乎真值）
        return { pw: m.solved > 0 ? 1 : 0,
                 ac: m.bestAction != null ? await areaAfter(state, p, m.bestAction) : null,
                 solved: m.solved, searched: true };
    }
    const dist = m.rootDist || [];
    const tot = dist.reduce((s, x) => s + x.n, 0);
    if (!tot) return null;
    let vWin = 0;
    for (const x of dist) vWin += (x.n / tot) * x.q;
    return { pw: (vWin + 1) / 2, searched: true };
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
    // 首选恒出在冠军那份权重上（复盘答的是"该怎么走"，与你这局玩的是哪档无关），
    // 所以这里不能跟着 play 走；只点过大师的用户第一次复盘会在这一步把冠军下下来。
    const champ = await ensurePlay('tongxuan');
    const r = await NN.chooseAction(state, player, {
        net: champ, budgetMs: REPLAY.budgetMs, simsMax: REPLAY.simsMax, cPuct: REPLAY.cPuct, topK: REPLAY.topK
    });
    if (!r) { send({ type: 'suggestion', id, error: '无合法着法' }); return; }
    // 前三选与读数同一份权重（冠军）⇒ 本手这一搜顺带出的 rootV 就是读数，口径天然一致（见 netReadouts）。
    // 本手这一搜顺带出的读数：主线程立刻拿去画形势条，不用等整局判读跑到这一帧。
    // cfg 是给面板标题报口径用的（sims 上限/PUCT 常数/预算），规则判定帧也带上
    const rd = await netReadouts(state, player, r.meta && r.meta.rootV);
    r.meta = Object.assign({}, r.meta, rd, { cfg: { simsMax: REPLAY.simsMax, budgetMs: REPLAY.budgetMs, cPuct: REPLAY.cPuct } });
    send({ type: 'suggestion', id, value: { moves: r.moves, wallDir: r.wallDir }, meta: r.meta });
}

self.onmessage = function (e) {
    const { state, id, ping, warmup, analyze, searchAnalyze, suggest, states, play } = e.data || {};
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
        const msg = (PLAY[play] ? PLAY[play].name : NN_NAME) + ' 不可用（已初始化失败）';
        send({ type: analyze ? 'analysis' : searchAnalyze ? 'searchAnalysis' : suggest ? 'suggestion' : 'error', id, message: msg, error: msg });
        return;
    }
    if (!initPromise) {
        // 首条消息是哪一档，init 就先把那一档的权重取下来（复盘/suggest 那一路 job 内部还会再 ensurePlay('tongxuan')）
        initPromise = init(play).catch(err => {
            initFailed = true;
            const message = (err && err.message) || String(err);
            send({ type: 'status', phase: 'failed', message });
            throw err;
        });
    }
    const job = analyze ? () => runAnalyze(states, id)
              : searchAnalyze ? () => runSearchAnalyze(states, id)
              : suggest ? () => runSuggest(state, id)
              // warmup = "把这档的权重先取下来"：换难度时页面会再发一次，init 已经加载过就是空操作
              : warmup ? () => ensurePlay(play)
              : state ? () => think(state, id, play)
              : null;
    if (!job) return;
    initPromise.then(job).catch(err => {
        const message = (err && err.message) || String(err);
        send({ type: analyze ? 'analysis' : searchAnalyze ? 'searchAnalysis' : suggest ? 'suggestion' : 'error', id, message, error: message });
    });
};
