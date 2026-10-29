/*
 * 围城 Siege 共享规则引擎 (SSOT)
 * 前端 Worker / 数据生成 / Arena 对拍 的唯一规则真源。
 * UMD: Node 用 require()，浏览器/importScripts 用 self.SiegeEngine。
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.SiegeEngine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const VERSION = '1.0.0';
    const BOARD_SIZE = 7;
    const N_CELLS = BOARD_SIZE * BOARD_SIZE;
    const MAX_STEPS = 3;
    const WALL_DIRS = ['top', 'right', 'bottom', 'left'];
    const DR = { top: -1, right: 0, bottom: 1, left: 0 };
    const DC = { top: 0, right: 1, bottom: 0, left: -1 };
    const OPP_DIR = { top: 'bottom', bottom: 'top', right: 'left', left: 'right' };
    const NUM_ACTIONS = N_CELLS * 4; // 196
    const NUM_PLANES = 10;

    function inBounds(r, c) { return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE; }

    function newGame() {
        const board = [];
        for (let r = 0; r < BOARD_SIZE; r++) {
            const row = [];
            for (let c = 0; c < BOARD_SIZE; c++) {
                row.push({ walls: { top: false, right: false, bottom: false, left: false } });
            }
            board.push(row);
        }
        return {
            board,
            players: [
                { row: BOARD_SIZE - 1, col: 0 },
                { row: 0, col: BOARD_SIZE - 1 }
            ],
            currentPlayer: 0,
            turnCount: 1
        };
    }

    function cloneState(state) {
        const board = new Array(BOARD_SIZE);
        for (let r = 0; r < BOARD_SIZE; r++) {
            const row = new Array(BOARD_SIZE);
            for (let c = 0; c < BOARD_SIZE; c++) {
                const w = state.board[r][c].walls;
                row[c] = { walls: { top: w.top, right: w.right, bottom: w.bottom, left: w.left } };
            }
            board[r] = row;
        }
        return {
            board,
            players: [
                { row: state.players[0].row, col: state.players[0].col },
                { row: state.players[1].row, col: state.players[1].col }
            ],
            currentPlayer: state.currentPlayer,
            turnCount: state.turnCount
        };
    }

    // 与旧内嵌引擎 getNeighborsFromState 逐行等价
    function neighbors(state, r, c) {
        const result = [];
        const cell = state.board[r][c];
        for (const d of WALL_DIRS) {
            const dr = DR[d], dc = DC[d];
            const nr = r + dr, nc = c + dc;
            if (!inBounds(nr, nc)) continue;
            if (cell.walls[d]) continue;
            if (state.board[nr][nc].walls[OPP_DIR[d]]) continue;
            result.push({ row: nr, col: nc });
        }
        return result;
    }

    // BFS <= MAX_STEPS 可达格（含起点），排除对手所在格。与旧 getReachableCells/aiSearch 一致
    function reachable(state, player) {
        const opp = 1 - player;
        const oppR = state.players[opp].row, oppC = state.players[opp].col;
        const start = state.players[player];
        const visited = new Set([start.row * BOARD_SIZE + start.col]);
        const queue = [{ row: start.row, col: start.col, steps: 0 }];
        const out = [];
        while (queue.length > 0) {
            const cur = queue.shift();
            out.push({ row: cur.row, col: cur.col, steps: cur.steps });
            if (cur.steps >= MAX_STEPS) continue;
            for (const nb of neighbors(state, cur.row, cur.col)) {
                if (nb.row === oppR && nb.col === oppC) continue;
                const k = nb.row * BOARD_SIZE + nb.col;
                if (!visited.has(k)) {
                    visited.add(k);
                    queue.push({ row: nb.row, col: nb.col, steps: cur.steps + 1 });
                }
            }
        }
        return out;
    }

    function actionId(r, c, dirIdx) { return (r * BOARD_SIZE + c) * 4 + dirIdx; }
    function actionDecode(id) {
        const cell = Math.floor(id / 4);
        return { row: Math.floor(cell / BOARD_SIZE), col: cell % BOARD_SIZE, dirIdx: id % 4 };
    }

    // 边合法性：非棋盘外、两边均无墙（与旧 selectWallDir/confirmWall 校验一致）
    function canPlaceWall(state, r, c, dirIdx) {
        const dir = WALL_DIRS[dirIdx];
        if (state.board[r][c].walls[dir]) return false;
        const nr = r + DR[dir], nc = c + DC[dir];
        if (!inBounds(nr, nc)) return false;
        if (state.board[nr][nc].walls[OPP_DIR[dir]]) return false;
        return true;
    }

    // 复合动作 = 落点(可达) + 从落点放一道墙。与旧 aiSearch actions 枚举一致。
    // 若整个可达域内无任何可放墙位（棋盘饱和），返回空数组。
    function legalActions(state, player) {
        const targets = reachable(state, player);
        const ids = [];
        for (const t of targets) {
            for (let d = 0; d < 4; d++) {
                if (canPlaceWall(state, t.row, t.col, d)) ids.push(actionId(t.row, t.col, d));
            }
        }
        return ids;
    }

    // 放墙（两侧同步写入），与旧 applyWall 一致
    function placeWall(state, r, c, dirIdx) {
        const dir = WALL_DIRS[dirIdx];
        const nr = r + DR[dir], nc = c + DC[dir];
        state.board[r][c].walls[dir] = true;
        if (inBounds(nr, nc)) state.board[nr][nc].walls[OPP_DIR[dir]] = true;
    }

    // 执行动作并结束该回合（切换玩家、回合数在蓝方行动后+1，与旧 endTurn 一致）
    function applyAction(state, player, action) {
        const { row, col, dirIdx } = actionDecode(action);
        state.players[player].row = row;
        state.players[player].col = col;
        placeWall(state, row, col, dirIdx);
        state.currentPlayer = 1 - player;
        if (player === 1) state.turnCount++;
    }

    // 与旧 computeRegionsFromState 一致：未分隔时返回 [size0, size0, false]
    function computeRegions(state) {
        const flood = (sr, sc) => {
            const visited = new Set([sr * BOARD_SIZE + sc]);
            const queue = [{ row: sr, col: sc }];
            while (queue.length > 0) {
                const cur = queue.shift();
                for (const nb of neighbors(state, cur.row, cur.col)) {
                    const k = nb.row * BOARD_SIZE + nb.col;
                    if (!visited.has(k)) { visited.add(k); queue.push(nb); }
                }
            }
            return visited;
        };
        const v0 = flood(state.players[0].row, state.players[0].col);
        const s0 = v0.size;
        const sep = !v0.has(state.players[1].row * BOARD_SIZE + state.players[1].col);
        if (!sep) return [s0, s0, false];
        const v1 = flood(state.players[1].row, state.players[1].col);
        return [s0, v1.size, true];
    }

    // 终局：分隔即终局，面积大者胜，相同为平局
    function terminalInfo(state) {
        const [s0, s1, sep] = computeRegions(state);
        if (!sep) return null;
        return {
            s0, s1,
            winner: s0 > s1 ? 0 : s1 > s0 ? 1 : 'draw',
            reward: Math.sign(s0 - s1) // 从 player0 视角
        };
    }

    // 与旧 getStateKey 逐字节一致（54字节）
    function stateKey(state, perspective) {
        const buf = new Uint8Array(54);
        buf[0] = perspective;
        buf[1] = state.players[0].row; buf[2] = state.players[0].col;
        buf[3] = state.players[1].row; buf[4] = state.players[1].col;
        let i = 5;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const w = state.board[r][c].walls;
                buf[i++] = (w.top ? 1 : 0) | (w.right ? 2 : 0) | (w.bottom ? 4 : 0) | (w.left ? 8 : 0);
            }
        }
        return String.fromCharCode(...buf);
    }

    // 全 BFS 距离场（无视对手阻挡，与 computeRegions 同一套邻接）。返回 Int16Array(49)，不可达 -1
    function distMap(state, player) {
        const dist = new Int16Array(N_CELLS).fill(-1);
        const sr = state.players[player].row, sc = state.players[player].col;
        dist[sr * BOARD_SIZE + sc] = 0;
        const queue = [{ row: sr, col: sc }];
        while (queue.length > 0) {
            const cur = queue.shift();
            const d = dist[cur.row * BOARD_SIZE + cur.col];
            for (const nb of neighbors(state, cur.row, cur.col)) {
                const k = nb.row * BOARD_SIZE + nb.col;
                if (dist[k] < 0) { dist[k] = d + 1; queue.push(nb); }
            }
        }
        return dist;
    }

    /* 特征平面：10 × 49，perspective 玩家的视角
     * 0..3: 墙 top/right/bottom/left  4: 己方棋子  5: 对手棋子
     * 6: 己方3步可达  7: 对手3步可达  8: 当前连通域(=计分基数)
     * 9: 归属图 —— 每格按"谁 BFS 先到"判给己方(1)/中立(0.5)/对方(0)
     *
     * plane 9 换掉的是"对手连通域"，因为它与 plane 8 **恒等**：规则上分隔即终局，
     * 所以凡是被记录的局面双方都在同一连通域里，两个 flood 必然同一个集合
     * （实测 sp_gen003 33703/33703 行 plane8 与 plane9 逐字节相同 ⇒ 49 个平面纯浪费）。
     *
     * 实测这张归属图的增益**有限，别高估**（22,550 行 teacher+自博弈混样，与终局面积差的 Pearson r）：
     *   归属图计数差 0.275 ｜ 已有的「3步可达 我-敌」0.261 ｜ 二者一起 R=0.287
     * ⇒ 相比编码里本来就有的量，只多带来 +0.026 的相关，不是"钥匙"。改它的真正理由是
     *   ①免费（平面宽度不变、旧数据可从 plane0-5 无损重算，见 migrate_v3.js）；
     *   ②它给的是"哪些格有争议、争议多少"这种**逐格**结构，比 plane6/7 的 3 步截断在空间上更完整。
     * 更要紧的结论在别处：**手头最好的单特征也只能解释 r²≈7% 的终局面积差**，
     *   剩下的由此后 20+ 次落墙决定 ⇒ 中盘 value 的天花板本来就低，不要指望表征能救它。
     * 已知代价：中立格（同距）记 0.5，而它在终局必被一方拿走；实测对 r 的影响小于上面那个量级。 */
    function featurePlanes(state, perspective) {
        const out = new Float32Array(NUM_PLANES * N_CELLS);
        const opp = 1 - perspective;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const k = r * BOARD_SIZE + c;
                const w = state.board[r][c].walls;
                if (w.top) out[k] = 1;
                if (w.right) out[N_CELLS + k] = 1;
                if (w.bottom) out[2 * N_CELLS + k] = 1;
                if (w.left) out[3 * N_CELLS + k] = 1;
            }
        }
        out[4 * N_CELLS + state.players[perspective].row * BOARD_SIZE + state.players[perspective].col] = 1;
        out[5 * N_CELLS + state.players[opp].row * BOARD_SIZE + state.players[opp].col] = 1;
        const mark = (list, base) => {
            for (const t of list) out[base + t.row * BOARD_SIZE + t.col] = 1;
        };
        mark(reachable(state, perspective), 6 * N_CELLS);
        mark(reachable(state, opp), 7 * N_CELLS);
        const dMe = distMap(state, perspective), dOpp = distMap(state, opp);
        // plane 8 = 视角方所在连通域（未分隔时即全域，= 终局计分的基数；与 computeRegions 同一 flood）
        for (let k = 0; k < N_CELLS; k++) if (dMe[k] >= 0) out[8 * N_CELLS + k] = 1;
        for (let k = 0; k < N_CELLS; k++) {
            const a = dMe[k], b = dOpp[k];
            if (a < 0 && b < 0) continue;                       // 双方都到不了：已隔出，留 0
            if (b < 0) { out[9 * N_CELLS + k] = 1; continue; }   // 只有我到得了 → 我方
            if (a < 0) continue;                                 // 只有对方到得了 → 留 0（视角方的 0 即对方）
            out[9 * N_CELLS + k] = a < b ? 1 : (b < a ? 0 : 0.5);
        }
        return out;
    }

    // 合法性掩码（196维）
    function actionMask(state, player) {
        const mask = new Uint8Array(NUM_ACTIONS);
        for (const id of legalActions(state, player)) mask[id] = 1;
        return mask;
    }

    return {
        VERSION, BOARD_SIZE, N_CELLS, MAX_STEPS, WALL_DIRS, DR, DC, OPP_DIR,
        NUM_ACTIONS, NUM_PLANES,
        inBounds, newGame, cloneState, neighbors, reachable, distMap,
        actionId, actionDecode, canPlaceWall, legalActions,
        placeWall, applyAction, computeRegions, terminalInfo,
        stateKey, featurePlanes, actionMask
    };
});
