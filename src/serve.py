"""OpenAI 互換 API サーバの起動ラッパ。

`mlx_lm.server` をそのまま使うが、LFM2.5 向けに2点だけモンキーパッチで補う:

1. ツールコール有効化:
   mlx-lm の自動判定 `_infer_tool_parser` は `<|tool_list_start|>` を探すため、
   LFM2.5 の `<|tool_call_start|>` 形式を取りこぼし `has_tool_calling=False` になる。
   一方で対応パーサ (`pythonic`) は mlx-lm に同梱済みなので、テンプレートに
   `<|tool_call_start|>` が含まれるモデルでは pythonic を割り当てる。

2. KV キャッシュ量子化 (任意):
   `mlx_lm.server` は `stream_generate` に kv_bits を渡さない (=フル精度) ため、
   `--kv-bits` 指定時のみ stream_generate をラップして量子化を配線する。
   LFM2.5 は hybrid 構成なので、量子化は `to_quantized` を持つ attention 層の
   KV のみに適用され、recurrent/conv 層のキャッシュは自動的にスキップされる。

それ以外 (OpenAI 互換エンドポイント・SSE・プロンプト KV キャッシュ再利用・
投機デコード・並行制御) は mlx_lm.server の標準機能をそのまま使う。
"""

import functools
import sys

import mlx_lm.tokenizer_utils as tokenizer_utils
from mlx_lm import server

DEFAULT_MODEL = "LiquidAI/LFM2.5-8B-A1B-MLX-4bit"
TOOL_CALL_MARKER = "<|tool_call_start|>"

# LiquidAI 推奨サンプリング既定値 (リクエストが明示しない場合のみ適用)。
# repetition_penalty はサーバ CLI に無いので、必要ならクライアント側で指定する。
SAMPLING_DEFAULTS = {"--temp": "0.2", "--top-k": "80"}


def _patch_tool_parser() -> None:
    """LFM2.5 の pythonic ツールコール形式を自動検出に載せる。"""
    original = tokenizer_utils._infer_tool_parser

    @functools.wraps(original)
    def infer(chat_template):
        if isinstance(chat_template, str) and TOOL_CALL_MARKER in chat_template:
            return "pythonic"
        return original(chat_template)

    tokenizer_utils._infer_tool_parser = infer


def _patch_kv_quant(kv_bits: int, kv_group_size: int, quantized_kv_start: int) -> None:
    """stream_generate に KV キャッシュ量子化パラメータを注入する。"""
    # server が .generate から取り込んだ stream_generate を動的に差し替える。
    original = getattr(server, "stream_generate")

    @functools.wraps(original)
    def stream_generate(*args, **kwargs):
        kwargs.setdefault("kv_bits", kv_bits)
        kwargs.setdefault("kv_group_size", kv_group_size)
        kwargs.setdefault("quantized_kv_start", quantized_kv_start)
        return original(*args, **kwargs)

    setattr(server, "stream_generate", stream_generate)


def _pop_option(argv: list[str], name: str) -> str | None:
    """argv から `--name VALUE` / `--name=VALUE` を取り除き、値を返す (無ければ None)。"""
    for i, token in enumerate(argv):
        if token == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} に値がありません")
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            del argv[i]
            return value
    return None


def _has_option(argv: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in argv)


def main() -> None:
    argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        print(
            "lfm2-serve: mlx_lm.server のラッパ。追加オプション:\n"
            "  --kv-bits N            KV キャッシュ量子化のビット数 (例: 8 / 4。既定: 無効=フル精度)\n"
            "  --kv-group-size N      量子化グループサイズ (既定: 64)\n"
            "  --quantized-kv-start N このトークン数以降の KV を量子化 (既定: 0)\n"
            "以下は mlx_lm.server 本体のオプション:\n",
            file=sys.stderr,
        )

    # --- 追加オプションを抽出 (本体 argparse に渡す前に取り除く) ---
    kv_bits = _pop_option(argv, "--kv-bits")
    kv_group_size = _pop_option(argv, "--kv-group-size")
    quantized_kv_start = _pop_option(argv, "--quantized-kv-start")

    # --- パッチ適用 ---
    _patch_tool_parser()
    if kv_bits is not None:
        _patch_kv_quant(
            kv_bits=int(kv_bits),
            kv_group_size=int(kv_group_size) if kv_group_size else 64,
            quantized_kv_start=int(quantized_kv_start) if quantized_kv_start else 0,
        )

    # --- 既定値の補完 (ユーザ指定があればそちらを尊重) ---
    if not _has_option(argv, "--model"):
        argv += ["--model", DEFAULT_MODEL]
    for flag, value in SAMPLING_DEFAULTS.items():
        if not _has_option(argv, flag):
            argv += [flag, value]

    # mlx_lm.server.main() は sys.argv を読むので差し替えて委譲する。
    sys.argv = ["lfm2-serve", *argv]
    server.main()


if __name__ == "__main__":
    main()
