# my-LFM2.5-agent

LiquidAI の **LFM2.5-8B-A1B** (総 8.3B / アクティブ 1.5B の hybrid MoE) を
[mlx-lm](https://github.com/ml-explore/mlx-lm) を推論エンジンに、Apple Silicon
上で動かすための最小セットアップ。

## 必要環境

- Apple Silicon Mac (M1 以降)
- Python 3.14 以上
- [uv](https://docs.astral.sh/uv/) (`flake.nix` の devShell 内なら自動で利用可能)
- 4bit モデルでディスク ~4.5GB / ピーク RAM ~5GB

## セットアップ

```bash
# Nix を使う場合
nix develop          # uv / ruff / pyright が入った shell に入る

# 依存をインストール (.venv が作られる)
uv sync
```

初回 `src/run.py` 実行時に Hugging Face Hub から MLX 変換済みモデルが
`~/.cache/huggingface/` にダウンロードされる (4bit で ~4.5GB)。

## 使い方

### 単発プロンプト (ストリーミング)

```bash
.venv/bin/python src/run.py --prompt "C. elegans を1文で説明して。"
```

### 対話モード

引数 `--prompt` を省略するとチャットになる。`/reset` で履歴クリア、
`Ctrl-C` で入力中の行をキャンセル、`Ctrl-D` で終了。入力は
[prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) 経由なので、
日本語 (全角・IME 変換) でもカーソル位置が崩れず、矢印キー編集と永続履歴
(`~/.cache/lfm2-agent/chat_history`) が使える。

```bash
.venv/bin/python src/run.py
```

### 量子化バリアントの切り替え

LiquidAI が公式に 4bit / 5bit / 6bit / 8bit / bf16 を配布しているので、
`--model` で差し替え可能。

```bash
.venv/bin/python src/run.py --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit
```

| variant | ディスク容量 | 用途 |
| --- | --- | --- |
| `LFM2.5-8B-A1B-MLX-4bit` (デフォルト) | ~4.5GB | 速度・メモリ最優先 |
| `LFM2.5-8B-A1B-MLX-6bit` | ~6.5GB | バランス |
| `LFM2.5-8B-A1B-MLX-8bit` | ~8.5GB | ほぼ無損失 |
| `LFM2.5-8B-A1B-MLX-bf16` | ~16GB  | フル精度 |

### サンプリング・パラメータ

デフォルトは LiquidAI 推奨の `temperature=0.2`, `top_k=80`,
`repetition_penalty=1.05`。Reasoning 出力 (`<think>` トレース) が長いので、
最終回答まで欲しいときは `--max-tokens` を 2048 以上にすると安定する。

```bash
.venv/bin/python src/run.py \
  --prompt "東京から京都の最短ルートを考えて。" \
  --max-tokens 2048 \
  --temperature 0.2 \
  --top-k 80 \
  --repetition-penalty 1.05
```

### 全オプション

```bash
.venv/bin/python src/run.py --help
```

## エージェント向け: OpenAI 互換 API サーバ

Hermes / コーディングエージェントなどから使う場合は、対話 CLI ではなく
`src/serve.py` で OpenAI 互換サーバを立てる。中身は `mlx_lm.server` で、
ツールコール・SSE ストリーミング・プロンプト KV キャッシュ再利用・投機デコードを
内蔵している。

```bash
.venv/bin/python src/serve.py --port 8080
# → http://localhost:8080/v1 が OpenAI 互換エンドポイントになる
```

エージェント側の `OPENAI_BASE_URL` に `http://localhost:8080/v1` を、
`OPENAI_API_KEY` には任意の文字列を設定すれば接続できる。

### なぜラッパが要るか

`mlx_lm.server` 単体だと LFM2.5 のツールコールが無効になる。mlx-lm の自動判定が
`<|tool_list_start|>` を探すのに対し、LFM2.5 は `<|tool_call_start|>` 形式
(Pythonic: `[get_weather(city="Tokyo", days=3)]`) を使うため取りこぼすのが原因。
`src/serve.py` は、対応パーサ (mlx-lm 同梱の `pythonic`) を割り当てる
モンキーパッチを当てて、これを OpenAI 形式の `tool_calls` JSON に変換する。
あわせて LiquidAI 推奨サンプリング (`temp=0.2`, `top_k=80`) を既定にしている。

```bash
# ツールコール疎通確認 (tool_calls が返れば成功)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
       "messages":[{"role":"user","content":"What is the weather in Tokyo for 3 days?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "parameters":{"type":"object","properties":{
           "city":{"type":"string"},"days":{"type":"integer"}},
           "required":["city","days"]}}}]}'
```

### KV キャッシュ量子化 (任意)

`mlx_lm.server` は KV キャッシュをフル精度で持つ (量子化フラグが無い)。
`src/serve.py` は `--kv-bits` 指定時のみ KV 量子化を配線する。

```bash
.venv/bin/python src/serve.py --port 8080 --kv-bits 8   # 8bit (4bit も可)
```

ただし LFM2.5 は hybrid 構成で、24 層中 KV を持つ attention 層は 6 つだけ
(残り 18 は recurrent/conv 層で量子化対象外・自動スキップ)。よって省メモリ効果は
限定的で、**長コンテキストのエージェントセッションでメモリが逼迫するとき**に
有効化する用途。通常は無効 (フル精度) のままで良い。

| 追加オプション | 既定 | 説明 |
| --- | --- | --- |
| `--kv-bits N` | 無効 | KV 量子化のビット数 (8 / 4)。未指定ならフル精度 |
| `--kv-group-size N` | 64 | 量子化グループサイズ |
| `--quantized-kv-start N` | 0 | このトークン数以降の KV を量子化 |

そのほか `--draft-model`(投機デコード)、`--prompt-cache-size`、
`--decode-concurrency` など `mlx_lm.server` 本体のオプションがそのまま使える。

## mlx-lm 同梱 CLI を使う

`src/run.py` を介さずに、mlx-lm 同梱の CLI から直接呼ぶこともできる。

```bash
# 単発生成
.venv/bin/mlx_lm.generate \
  --model LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  --prompt "Hello" \
  --max-tokens 512 \
  --temp 0.2 --top-k 80

# 対話
.venv/bin/mlx_lm.chat --model LiquidAI/LFM2.5-8B-A1B-MLX-4bit
```

ただし chat template や repetition_penalty などの推奨値は手で渡す必要が
あるので、通常は `src/run.py` 経由を推奨。
