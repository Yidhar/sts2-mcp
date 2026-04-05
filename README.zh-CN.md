# sts2-mcp 🌌

[![Chinese](https://img.shields.io/badge/lang-中文-red.svg)](#) [![English](https://img.shields.io/badge/lang-English-blue.svg)](./README.md)

`sts2-mcp` 是一个为 **《杀戮尖塔 2》(Slay the Spire 2)** 打造的高性能本地控制栈。它利用 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 协议，在游戏的内部状态与外部 AI Agent 之间搭建了一座桥梁。

与传统的截图识别 (Screen-scraping) 或 OCR 方案不同，`sts2-mcp` 通过一个**原生 C# 桥接模组 (Bridge Mod)** 直接提取精确的游戏数据并暴露合法动作，确保了 100% 的准确性和毫秒级的响应延迟。

---

## 🏗️ 项目架构

项目主要由两个核心组件组成：

### 1. `mods/sts2-bridge` (传感器与执行器)
一个直接注入到《杀戮尖塔 2》进程中的 C#/.NET 9 模组。
- **状态序列化**：将复杂的内部游戏对象（运行数据、战斗、奖励、地图等）转换为干净的 JSON。
- **动作执行**：直接调用游戏方法执行出牌、选择和地图移动。
- **自动发现**：自动创建 session 文件，方便 MCP 服务定位并连接。

### 2. `packages/mcp-server` (交互界面)
基于 Node.js 22 的标准 MCP 协议实现。
- **工具映射**：将桥接器的 HTTP 接口转换为标准的 MCP Tools。
- **操作安全**：引入 `state_version` 保护机制，防止执行“过时”动作（例如尝试打出一张已经消耗掉的牌）。
- **流程优化**：针对奖励、商店、营火等复杂场景实现批量化处理，大幅减少 LLM 的往返调用。

---

## 🛠️ MCP 工具参考 (Tools)

该服务向任何兼容 MCP 的 Agent（如 Claude Desktop 或 Antigravity）暴露以下工具：

| 工具名称 | 功能描述 |
| :--- | :--- |
| `sts2_get_state` | 获取完整的当前游戏状态（界面、血量、卡组、遗物等）。 |
| `sts2_get_deck` | 返回完整主牌组，适合商店、升级和 Boss 前的低频策略判断。 |
| `sts2_list_actions` | 列出当前玩家所有合法的可选动作。 |
| `sts2_perform_action` | 通过唯一的 Action ID 执行单个动作，并可通过 `return_state_after` 一并返回完整的后置原始状态。 |
| `sts2_play_card_sequence` | 按顺序打出多张卡牌，并自动处理手牌索引重排。 |
| `sts2_execute_combat_sequence` | 在一次调用中混合执行出牌、喝药和 `end_turn`，并自动处理重匹配。 |
| `sts2_resolve_room_rewards` | 一键领取金币/药水，并可选择性地拿取卡牌奖励。 |
| `sts2_resolve_rest_site` | 执行休息/锻造操作，并自动返回地图。 |
| `sts2_resolve_card_selection` | 完美处理选牌、跳过或转化的弹出界面。 |
| `sts2_pick_option` | 以统一索引方式选择奖励/事件/营火/选牌项，不再依赖原始 Action ID。 |
| `sts2_travel_to_coordinate` | 自动吸收奖励/营火收尾并等待地图稳定后，再移动到指定坐标。 |
| `sts2_resolve_shop_visit` | 在单次批量操作中购买多个物品并执行移除卡牌。 |

---

## 🚀 快速上手

### 环境要求
- **操作系统**: Windows (目前《杀戮尖塔 2》仅支持 Windows)。
- **运行环境**: [Node.js 22+](https://nodejs.org/)。
- **游戏**: 已安装的正版《杀戮尖塔 2》。

### 选项 A: 使用预编译版本 (推荐)
1. 从 Releases 页面下载最新的 `sts2-bridge` 预编译包 (包含 DLL 等文件)。
2. 在游戏的 `mods` 目录下创建一个名为 `sts2-bridge` 的文件夹。
   - 例如: `<杀戮尖塔2安装目录>\mods\sts2-bridge`
3. 将下载的文件放入该文件夹中。

### 选项 B: 从源码编译
如果你想自行编译桥接模组，你还需要安装 [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)。
设置你的游戏安装路径并构建项目：

```powershell
$env:STS2_DIR = "<杀戮尖塔2安装目录>"
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```
*注意：构建脚本会自动将生成的文件拷贝到游戏的 `mods\sts2-bridge` 文件夹中。*

### 后续步骤...

1. **启动游戏**: 运行《杀戮尖塔 2》。桥接模组将初始化并在 `%APPDATA%\SlayTheSpire2\bridge\session.json` 生成会话文件。
2. **启动 MCP 服务**:
```powershell
node .\packages\mcp-server\index.js
```

---

## 🗺️ 路线图 & TODO

- [ ] **更完善的 Tools**: 扩展 MCP 工具集，支持更细粒度的状态查询和复杂的动作序列。
- [ ] **联机支持**: 允许 Agent 交互或管理游戏内的联机/合作模式机制。
- [ ] **游戏内对话注入**: 将 AI 的策略文本和思考过程直接作为对话框内容注入到游戏中，提升交互的沉浸感。

---

## 📝 配置说明

MCP 服务会自动寻找桥接会话文件。你可以通过环境变量手动覆盖：
- `STS2_BRIDGE_SESSION_FILE`: 指向自定义会话 JSON 的路径。

战斗调用说明：
- 不要把连续出牌拆成并行的 `sts2_perform_action`。
- 纯出牌回合优先使用 `sts2_play_card_sequence`。
- 如果同一回合要混合出牌、喝药或结束回合，优先使用 `sts2_execute_combat_sequence`。

---

## 🤖 统一 RL 流水线

RL 分支现在使用同一条 checkpoint 链路贯穿三个阶段：

1. 战斗沙盒 PPO
2. 离线构筑 / 路线预训练
3. 全流程 PPO

正式入口是 [packages/rl-agent/train_pipeline.py](./packages/rl-agent/train_pipeline.py)。它会把每一阶段产出的 checkpoint 自动传给下一阶段。

示例：

```powershell
python .\packages\rl-agent\train_pipeline.py `
  --dataset-root .\datasets\parquet `
  --character ironclad `
  --session-file "$env:APPDATA\SlayTheSpire2\bridge\session.json" `
  --stage2-partition-kind build_family `
  --stage2-partition-value v0.98_to_v0.99.1
```

如果不传 `--stage1-encounter-pool`，第一阶段现在会默认使用一组更贴近初始牌组难度的一幕前几层弱怪池。这个集合是根据已导出的 run 历史统计出来的：

- `ENCOUNTER.SLIMES_WEAK`
- `ENCOUNTER.SHRINKER_BEETLE_WEAK`
- `ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK`
- `ENCOUNTER.NIBBITS_WEAK`

固定评估的 holdout 池也默认切到未参与训练的一幕前几层弱怪：

- `ENCOUNTER.CORPSE_SLUGS_WEAK`
- `ENCOUNTER.SLUDGE_SPINNER_WEAK`
- `ENCOUNTER.SEAPUNK_WEAK`
- `ENCOUNTER.TOADPOLES_WEAK`

如果你已经有战斗沙盒 checkpoint，想直接跳过第一阶段：

```powershell
python .\packages\rl-agent\train_pipeline.py `
  --dataset-root .\datasets\parquet `
  --start-checkpoint .\pipeline_runs\some_run\stage1_sandbox\checkpoints\final `
  --stop-after offline
```

---

## ⚖️ 免责声明与许可证

**免责声明**: 这是一个非官方的社区项目。它与 Mega Crit 或《杀戮尖塔 2》的开发者没有隶属关系、背书或关联。使用风险自负。

**许可证**: [MIT](./LICENSE)
