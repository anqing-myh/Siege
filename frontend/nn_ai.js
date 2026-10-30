/*
 * 前端神经网络 AI (P4)：浏览器 Worker 内使用的 PUCT + ONNX 推理模块。
 * UMD：Worker 里 importScripts 后挂到 self.NnAI，Node 里 require() 用于离线测试。
 * 依赖：SiegeEngine（engine/siege_engine.js，UMD，需先加载）。
 *
 * 设计要点：
 *  - 输出格式与旧 AI 完全一致：{ moves: [{row,col}...], wallDir: 'top'|'right'|'bottom'|'left' }
 *    => index.html 只需多挂一个 Worker，替换结果来源，动画/校验/联机同步逻辑无需改动。
 *  - 时间预算驱动而非固定模拟数：每步给 budgetMs，模拟数随设备性能自适应。
 *  - 安全网：(a) 己方一步制胜的着法直接采纳，并在多个致胜着法中取面积差最大者（扩大面积零风险）；
 *    (b) 剔除"走完立即被隔离且己方区域更小"的自困着法；(c) 剔除"对手下一步可制胜"的送杀着法；
 *    (d) 若所有着法都输，则不再搜索，直接选"己方最终区域最大"的着法（保地盘/翻盘资本，不改变胜负）。
 *    注：绝不主动推迟已到手的胜利——多给对手一手的风险真实，而面积差不影响胜负。
 *  - 推理失败/超时/模型缺失 -> 调用方回退旧 AI Worker（本模块只负责抛错）。
 */
(function (root, factory) {
    const E = (typeof require === 'function' && typeof window === 'undefined')
        ? require('../engine/siege_engine.js')
        : (root.SiegeEngine || (typeof importScripts === 'function' ? self.SiegeEngine : null));
    const api = factory(E);
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.NnAI = api;
})(typeof self !== 'undefined' ? self : this, function (E) {
    'use strict';
    if (!E) throw new Error('NnAI 需要 SiegeEngine');

    const N = 196;

    // ---------- 神经网络句柄 ----------
    // ortNet: { session, inName, outNames } 由调用方用 onnxruntime-web 创建
    function createOrtNet(ort, session) {
        const inName = session.inputNames[0];
        const outNames = session.outputNames;
        const scratch = new Float32Array(E.NUM_PLANES * E.N_CELLS);
        return {
            async evaluate(state, player) {
                const planes = E.featurePlanes(state, player);
                scratch.set(planes);
                const feeds = {};
                feeds[inName] = new ort.Tensor('float32', scratch, [1, E.NUM_PLANES, E.BOARD_SIZE, E.BOARD_SIZE]);
                const res = await session.run(feeds);
                const logits = res[outNames[0]].data;
                const value = res[outNames[1]].data[0];
                const legal = E.legalActions(state, player);
                const mask = new Uint8Array(N); for (const a of legal) mask[a] = 1;
                return { probs: softmaxMasked(logits, mask, legal), mask, legal, v: value };
            }
        };
    }

    // 测试/降级用：均匀先验 + 手工 value（等价于旧评估函数，无网络时也能跑通链路）
    function createStubNet(valueFn) {
        return {
            async evaluate(state, player) {
                const legal = E.legalActions(state, player);
                const mask = new Uint8Array(N); for (const a of legal) mask[a] = 1;
                const p = 1 / Math.max(1, legal.length);
                const probs = new Float32Array(N); for (const a of legal) probs[a] = p;
                return { probs, mask, legal, v: Math.tanh(valueFn(state, player) / 300) };
            }
        };
    }

    function softmaxMasked(logits, mask, legal) {
        const probs = new Float32Array(N);
        let mx = -Infinity, sum = 0;
        for (const a of legal) if (logits[a] > mx) mx = logits[a];
        if (!isFinite(mx)) { for (const a of legal) probs[a] = 1 / legal.length; return probs; }
        for (const a of legal) { const x = Math.exp(logits[a] - mx); probs[a] = x; sum += x; }
        if (sum <= 0) { for (const a of legal) probs[a] = 1 / legal.length; return probs; }
        for (const a of legal) probs[a] /= sum;
        return probs;
    }

    // ---------- PUCT ----------
    function Node(state, player) {
        return { state, player, kids: new Map(), N: 0, W: 0, priors: null, legal: null, ready: false };
    }

    async function expand(node, net) {
        const r = await net.evaluate(node.state, node.player);
        node.legal = r.legal;
        node.priors = r.probs;
        node.ready = true;
        return r.v;
    }

    /*
     * 终局样本计数：复盘的"预测胜率"只认这个。
     * 只有真的走到终局的 sim 才是胜负样本 —— 预算耗尽停在半盘的 frontier 叶子给的是网络 v，
     * 那是倾向不是结果，混进去等于自己骗自己。计数按根行棋方视角，和=0.5 由调用方折算。
     */
    function tallyTerminal(cnt, rootPlayer, node, v) {
        if (!cnt) return;
        cnt.tn++;
        const rv = node.player === rootPlayer ? v : -v;
        if (rv > 0) cnt.tw++;
        else if (rv === 0) cnt.td++;
    }

    async function traverse(node, net, cPuct, cnt, rootPlayer) {
        if (rootPlayer === undefined) rootPlayer = node.player;   // 首次调用即根节点
        const t = E.terminalInfo(node.state);
        if (t) {
            // 终局叶子也必须计数。以前直接 return，这个孩子的 N 永远是 0 ⇒ Q 恒为 0（明明是精确 ±1），
            // 探索项 c·p·√N父/(1+0) 永不衰减 ⇒ 终局着法吸光 sims 却在访问分布里显示 0 次。
            // 前端另有 findImmediateWin 兜底，所以"漏赢"没外露，但中盘的 sims 一直在被白烧。
            const tv = node.player === 0 ? t.reward : -t.reward;
            node.N++; node.W += tv;
            tallyTerminal(cnt, rootPlayer, node, tv);
            return tv;
        }
        if (!node.ready) return await expand(node, net);
        if (!node.legal.length) return 0;

        let pick = -1, bestU = -Infinity;
        for (const a of node.legal) {
            const ch = node.kids.get(a);
            if (!ch) { pick = a; break; }
            const u = -ch.W / (ch.N || 1) + cPuct * node.priors[a] * Math.sqrt(node.N) / (1 + ch.N);
            if (u > bestU) { bestU = u; pick = a; }
        }
        let ch = node.kids.get(pick);
        if (!ch) {
            ch = Node(null, 1 - node.player);
            node.kids.set(pick, ch);
        }
        if (!ch.state) {
            ch.state = E.cloneState(node.state);
            E.applyAction(ch.state, node.player, pick);
        }
        const vn = -(await traverse(ch, net, cPuct, cnt, rootPlayer));
        node.W += vn; node.N++;
        return vn;
    }

    // 迭代版 traverse（避免深树 await 链开销）
    async function traverseIter(root, net, cPuct, cnt) {
        const path = [];
        let node = root, v;
        for (;;) {
            const t = E.terminalInfo(node.state);
            if (t) {
                v = node.player === 0 ? t.reward : -t.reward;
                // path 里存的是祖先节点，回传循环不会碰这个叶子 ⇒ 终局计数要在这里补（同 traverse）
                node.N++; node.W += v;
                tallyTerminal(cnt, root.player, node, v);
                break;
            }
            if (!node.ready) { v = await expand(node, net); break; }
            if (!node.legal.length) { v = 0; break; }
            let pick = -1, bestU = -Infinity;
            for (const a of node.legal) {
                const ch = node.kids.get(a);
                if (!ch) { pick = a; break; }
                const u = -ch.W / (ch.N || 1) + cPuct * node.priors[a] * Math.sqrt(node.N) / (1 + ch.N);
                if (u > bestU) { bestU = u; pick = a; }
            }
            let ch = node.kids.get(pick);
            if (!ch) { ch = Node(null, 1 - node.player); node.kids.set(pick, ch); }
            if (!ch.state) {
                ch.state = E.cloneState(node.state);
                E.applyAction(ch.state, node.player, pick);
            }
            path.push([node, pick]);
            node = ch;
        }
        for (let i = path.length - 1; i >= 0; i--) {
            const [nd] = path[i];
            v = -v;
            nd.W += v; nd.N++;
        }
        return v;
    }

    // ---------- 安全网 ----------
    // 走完后是否把自己关进更小的区域
    function selfTrapped(state, player, action) {
        const s = E.cloneState(state);
        E.applyAction(s, player, action);
        const t = E.terminalInfo(s);
        if (!t) return false;
        const mine = player === 0 ? t.s0 : t.s1;
        const theirs = player === 0 ? t.s1 : t.s0;
        return mine < theirs;
    }

    // 走完后对手能否一步制胜（返回对手能取得的最大面积差符号判断）
    function oppCanWinNext(state, player, action) {
        const s = E.cloneState(state);
        E.applyAction(s, player, action);
        const t0 = E.terminalInfo(s);
        if (t0) return t0.winner !== player; // 已经终局且不是自己赢
        const opp = 1 - player;
        for (const a of E.legalActions(s, opp)) {
            const s2 = E.cloneState(s);
            E.applyAction(s2, opp, a);
            const t = E.terminalInfo(s2);
            if (t && t.winner === opp) return true;
        }
        return false;
    }

    // 自己一步制胜：在所有致胜着法中取面积差最大者（结果已确定，扩大面积零风险）
    function findImmediateWin(state, player, legal) {
        let best = null;
        for (const a of legal) {
            const s = E.cloneState(state);
            E.applyAction(s, player, a);
            const t = E.terminalInfo(s);
            if (t && t.winner === player) {
                const margin = Math.abs(t.s0 - t.s1);
                if (!best || margin > best.margin) best = { action: a, margin };
            }
        }
        return best;
    }

    /*
     * 必败着法的结局估计：走完 action 后，若已分隔则直接取己方区域；
     * 否则假设对手从"能立即制胜"的着法中挑对自己最狠的一个，取己方届时区域。
     * 用于全灭分支"输得越少越好"的确定性排序（不改变胜负，只保住翻盘资本与地盘）。
     */
    function outcomeAreaIfLost(state, player, action) {
        const s = E.cloneState(state);
        E.applyAction(s, player, action);
        const t = E.terminalInfo(s);
        const mineOf = (tt) => (player === 0 ? tt.s0 : tt.s1);
        const theirsOf = (tt) => (player === 0 ? tt.s1 : tt.s0);
        if (t) return mineOf(t);
        const opp = 1 - player;
        let worstMine = Infinity, fallback = -1;
        for (const a2 of E.legalActions(s, opp)) {
            const s2 = E.cloneState(s);
            E.applyAction(s2, opp, a2);
            const t2 = E.terminalInfo(s2);
            if (!t2 || t2.winner !== opp) continue;
            const mine = mineOf(t2);
            const theirs = theirsOf(t2);
            // 对手按自身利益最大化（其面积-我方面积）；对我方即为最坏结局
            if (theirs - mine >= fallback) { fallback = theirs - mine; worstMine = mine; }
        }
        if (fallback < 0) {
            // 对手无法立即终结（理论上不会进入此分支）：用当前连通块大小作保守代理
            const [sz, , sep] = E.computeRegions(s);
            return sep ? mineOf({ s0: sz, s1: 0 }) : sz;
        }
        return worstMine;
    }

    // ---------- 动作 -> 旧 AI 输出格式（moves 路径 + wallDir） ----------
    function shortestPath(state, player, tr, tc) {
        const pl = state.players[player];
        const opp = 1 - player;
        const startKey = pl.row * E.BOARD_SIZE + pl.col;
        const prev = new Int16Array(E.N_CELLS).fill(-2);
        prev[startKey] = -1;
        const q = [startKey];
        let head = 0;
        while (head < q.length) {
            const k = q[head++];
            if (k === tr * E.BOARD_SIZE + tc) break;
            const r = (k / E.BOARD_SIZE) | 0, c = k % E.BOARD_SIZE;
            for (const nb of E.neighbors(state, r, c)) {
                const nk = nb.row * E.BOARD_SIZE + nb.col;
                if (prev[nk] !== -2) continue;
                if (nb.row === state.players[opp].row && nb.col === state.players[opp].col) continue;
                prev[nk] = k;
                q.push(nk);
            }
        }
        const goal = tr * E.BOARD_SIZE + tc;
        if (prev[goal] === -2) return null;
        const cells = [];
        for (let k = goal; k !== -1; k = prev[k]) cells.push({ row: (k / E.BOARD_SIZE) | 0, col: k % E.BOARD_SIZE });
        cells.reverse();
        return cells.slice(1); // 去掉起点
    }

    function toResult(state, player, action) {
        const { row, col, dirIdx } = E.actionDecode(action);
        const path = shortestPath(state, player, row, col);
        if (!path) return null;
        return { moves: path, wallDir: E.WALL_DIRS[dirIdx] };
    }

    // 复盘用：把根节点的访问分布导出（Q 取 -ch.W/ch.N 才是根行棋方视角）。
    // 只按访问数降序、同访问保持 pool 原序 —— 与 chooseAction 选 best 的规则一致，所以全表第 0 行就是 bestAction。
    // 返回**全表**不截断：top-K 与整局搜索判读的访问加权 V_root 都用它，截断会把权重算漏。
    function topCandidates(root, pool, ev) {
        const arr = [];
        for (const a of pool) {
            const ch = root.kids.get(a);
            arr.push({ a, n: ch ? ch.N : 0, q: ch && ch.N ? -ch.W / ch.N : 0, p: ev.probs[a] });
        }
        arr.sort((x, y) => y.n - x.n);
        return arr;
    }

    /*
     * 随机推演胜率：从这个局面起双方随手走到底，统计行棋方赢的比例。
     * 为什么用它而不是用搜索树里的终局样本（实测，NOTES §42）：
     *  - 400 访问的树在中盘只走到 0–32 个终局，开局一个都没有，而且这几十局全挤在搜索挑中的最佳线路上
     *    ⇒ 动不动就报 0% 或 100%，样本小又带偏，正是"莫名其妙 100 胜率"的那个坑；
     *  - 推演是纯引擎运算、不碰网络，一帧 300 局实测 20–105ms，比同帧那次搜索便宜，且每帧样本都是满的。
     * 它答的问题是"这局面放任走下去谁会输"（形势），不是"模型会怎么下"（那看首选行的 Q 与访问）。
     * 一次遍历顺带把终局面积差也累计下来（terminalInfo 本来就算好了 s0/s1）⇒ 胜率与期望面积差同价、同一批样本。
     * 没在 cap 手内终局的推演整条丢弃 —— 算成和棋会把"没结果"偷偷记成"各半"。
     * cap 默认 84：7×7 的内部边一共 2×7×6=84 条，一手放一条且墙不会消失 ⇒ 84 手内必然分隔，
     * 这个上限在实际对局里碰不到，留着只是防止推演失控的护栏（同理"无合法着法"那条分支也到不了）。
     */
    function rolloutScore(state, n, cap) {
        const p = state.currentPlayer;
        const maxPly = cap || 84;
        let cnt = 0, w = 0, d = 0;
        for (let i = 0; i < n; i++) {
            const s = E.cloneState(state);
            let res = null;
            for (let k = 0; k < maxPly; k++) {
                const t = E.terminalInfo(s);
                if (t) { res = t; break; }
                const legal = E.legalActions(s, s.currentPlayer);
                if (!legal.length) break;
                E.applyAction(s, s.currentPlayer, legal[(Math.random() * legal.length) | 0]);
            }
            if (!res) continue;
            cnt++;
            if (res.winner === 'draw') w += 0.5;
            else if (res.winner === p) w += 1;
            d += p === 0 ? res.s0 - res.s1 : res.s1 - res.s0;
        }
        return { n: cnt, w, d };
    }

    // ---------- 对外主入口 ----------
    /*
     * opts: { net, budgetMs=2500, simsMax=4000, cPuct=1.5, topK=0 }
     * 返回 { moves, wallDir, meta:{ sims, ms, bestAction, tn, tw, td[, top, rootV][, solved][, why] } }；null 表示无合法动作。
     * meta.tn/tw/td = 本次搜索里真走到终局的模拟数、其中根行棋方胜/平的局数 ⇒ 复盘胜率 = (tw+0.5td)/tn。
     * meta.solved = ±1 时胜负由规则判定（一步制胜 / 全都输），跟样本无关。
     * topK>0 时主搜索分支额外给 meta.top（前 K 个着法的访问/Q/先验）；安全网分支只有 meta.why，
     * 因为那几手是硬规则定的，不是模型权衡出来的——给假 top 反而误导。
     */
    async function chooseAction(state, player, opts) {
        const t0 = Date.now();
        const budget = opts.budgetMs || 2500;
        const cPuct = opts.cPuct == null ? 1.5 : opts.cPuct;
        const simsMax = opts.simsMax || 4000;
        const legal = E.legalActions(state, player);
        if (!legal.length) return null;

        // 1) 一步制胜：直接采纳（面积差最大者）
        const win = findImmediateWin(state, player, legal);
        if (win) {
            const r = toResult(state, player, win.action);
            // solved=+1：胜负已被规则判定，不需要样本，前端按 100% 画且不做收缩
            if (r) return Object.assign(r, { meta: { sims: 0, ms: Date.now() - t0, bestAction: win.action, why: 'immediate-win', solved: 1, tn: 0, tw: 0, td: 0 } });
        }

        // 2) 候选集裁剪：排除"自困"和"送对手一步杀"的着法
        const safe = [];
        for (const a of legal) {
            if (legal.length > 1 && selfTrapped(state, player, a)) continue;
            if (legal.length > 1 && oppCanWinNext(state, player, a)) continue;
            safe.push(a);
        }
        if (!safe.length) {
            // 全灭：着法都输（己方立即被隔离，或对手下一步必杀）。此时搜索无意义，
            // 直接选"己方最终区域最大"的着法——不改变胜负，只保留最大地盘与翻盘资本。
            let pick = legal[0], bestArea = -1;
            for (const a of legal) {
                const area = outcomeAreaIfLost(state, player, a);
                if (area > bestArea) { bestArea = area; pick = a; }
            }
            const r = toResult(state, player, pick);
            // solved=-1：每个合法着法都输，这是规则判定不是估计
            if (r) return Object.assign(r, { meta: { sims: 0, ms: Date.now() - t0, bestAction: pick, why: 'lost-save-area', area: bestArea, solved: -1, tn: 0, tw: 0, td: 0 } });
        }
        const pool = safe;
        if (pool.length === 1) {
            const r = toResult(state, player, pool[0]);
            if (r) return Object.assign(r, { meta: { sims: 0, ms: Date.now() - t0, bestAction: pool[0], why: 'only-safe' } });
        }

        // 3) PUCT：预算驱动
        const root = Node(E.cloneState(state), player);
        const ev = await opts.net.evaluate(root.state, player);
        root.legal = ev.legal; root.priors = ev.probs; root.ready = true;
        const cnt = { tn: 0, tw: 0, td: 0 };
        let sims = 0;
        while (sims < simsMax && Date.now() - t0 < budget * 0.8) {
            await traverseIter(root, opts.net, cPuct, cnt);
            sims++;
        }
        let best = pool[0], bestN = -1;
        for (const a of pool) {
            const ch = root.kids.get(a);
            const n = ch ? ch.N : -1;
            if (n > bestN) { bestN = n; best = a; }
        }
        const r = toResult(state, player, best);
        if (!r) { // 理论上不该发生：路径重建失败
            const fb = toResult(state, player, pool[0]);
            if (fb) return Object.assign(fb, { meta: { sims, ms: Date.now() - t0, bestAction: pool[0], why: 'pathfix', tn: cnt.tn, tw: cnt.tw, td: cnt.td } });
            return null;
        }
        r.meta = { sims, ms: Date.now() - t0, bestAction: best, excluded: legal.length - safe.length,
                   tn: cnt.tn, tw: cnt.tw, td: cnt.td };
        const topK = opts.topK || 0;
        if (topK > 0 || opts.rootDist) {
            const full = topCandidates(root, pool, ev);
            if (topK > 0) r.meta.top = full.slice(0, topK);
            // rootDist = 有访问的根候选全表：整局搜索判读拿它算访问加权的 V_root
            if (opts.rootDist) r.meta.rootDist = full.filter(x => x.n > 0);
            r.meta.rootV = ev.v;
        }
        return r;
    }

    return { createOrtNet, createStubNet, chooseAction, shortestPath, toResult, rolloutScore,
             selfTrapped, oppCanWinNext, outcomeAreaIfLost, findImmediateWin, traverse, traverseIter };
});
