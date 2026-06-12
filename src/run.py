"""LFM2.5-8B-A1B を mlx-lm で推論するエントリポイント。

デフォルトは LiquidAI 公式の 4bit MLX 変換版をロードし、初回実行時に
Hugging Face Hub からダウンロードされる (~/.cache/huggingface)。
"""

import os

import click
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

DEFAULT_MODEL = "LiquidAI/LFM2.5-8B-A1B-MLX-4bit"
DEFAULT_SYSTEM = "あなたは優秀なアルゴリズム設計者です。プログラムを生成する際には可読性と計算量を常に意識してください。"


def history_path() -> str:
    """対話履歴の保存先 (XDG_STATE_HOME があれば優先、無ければ ~/.cache)。"""
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.cache")
    directory = os.path.join(base, "lfm2-agent")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "chat_history")


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def generate_once(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_tokens: int,
    sampler,
    processors,
    stream: bool,
) -> str:
    pieces: list[str] = []
    last_response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        pieces.append(response.text)
        if stream:
            click.echo(response.text, nl=False)
        last_response = response
    if stream:
        click.echo()
    if last_response is not None:
        click.echo(
            f"[prompt {last_response.prompt_tokens} tok @ {last_response.prompt_tps:.1f} tok/s | "
            f"gen {last_response.generation_tokens} tok @ {last_response.generation_tps:.1f} tok/s | "
            f"peak {last_response.peak_memory:.2f} GB]",
            err=True,
        )
    return "".join(pieces)


def run_single(
    model, tokenizer, *, system: str, prompt: str, max_tokens: int, sampler, processors, stream: bool
) -> None:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    prompt_text = render_prompt(tokenizer, messages)
    text = generate_once(
        model,
        tokenizer,
        prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
        processors=processors,
        stream=stream,
    )
    if not stream:
        click.echo(text)


def run_chat(
    model, tokenizer, *, system: str, max_tokens: int, sampler, processors, stream: bool
) -> None:
    click.echo("対話モード (Ctrl-D で終了, Ctrl-C で入力キャンセル, /reset で履歴クリア)", err=True)
    # 入力は prompt_toolkit に任せる。macOS の libedit は全角を1カラム幅と
    # 誤算してカーソル位置・IME 変換中の再描画・行折り返しが崩れるため、
    # CJK 幅を wcwidth で正しく扱う prompt_toolkit に置き換えている。
    # ついでに矢印キー編集・永続履歴・ブラケットペーストも multibyte で安定する。
    session = PromptSession(history=FileHistory(history_path()))
    messages: list[dict] = [{"role": "system", "content": system}]
    while True:
        try:
            user_input = session.prompt("\nyou> ").strip()
        except EOFError:  # Ctrl-D で終了
            click.echo(err=True)
            return
        except KeyboardInterrupt:  # Ctrl-C は実行中の入力行だけ破棄して継続
            continue
        if not user_input:
            continue
        if user_input == "/reset":
            messages = [{"role": "system", "content": system}]
            click.echo("(履歴をクリアしました)", err=True)
            continue
        messages.append({"role": "user", "content": user_input})
        prompt_text = render_prompt(tokenizer, messages)
        click.echo("assistant> ", nl=False)
        reply = generate_once(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            processors=processors,
            stream=stream,
        )
        messages.append({"role": "assistant", "content": reply})


# LiquidAI 推奨: temperature=0.2, top_k=80, repetition_penalty=1.05
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "model_id", default=DEFAULT_MODEL, show_default=True,
              help="HF Hub のモデル ID もしくはローカルパス")
@click.option("--prompt", "prompt", default=None,
              help="単発プロンプト。未指定なら対話モード")
@click.option("--system", "system", default=DEFAULT_SYSTEM, show_default=True,
              help="システムプロンプト")
@click.option("--max-tokens", default=1024, show_default=True, type=int)
@click.option("--temperature", default=0.2, show_default=True, type=float)
@click.option("--top-k", default=80, show_default=True, type=int)
@click.option("--top-p", default=0.0, show_default=True, type=float)
@click.option("--repetition-penalty", default=1.05, show_default=True, type=float)
@click.option("--stream/--no-stream", default=True, show_default=True,
              help="ストリーミング出力の有効/無効")
def main(
    model_id: str,
    prompt: str | None,
    system: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stream: bool,
) -> None:
    """LFM2.5-8B-A1B を mlx-lm で実行する CLI。"""
    click.echo(f"loading model: {model_id}", err=True)
    loaded = load(model_id, return_config=False)
    model, tokenizer = loaded[0], loaded[1]
    sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
    processors = make_logits_processors(repetition_penalty=repetition_penalty)
    if prompt is not None:
        run_single(
            model, tokenizer,
            system=system, prompt=prompt, max_tokens=max_tokens,
            sampler=sampler, processors=processors, stream=stream,
        )
    else:
        run_chat(
            model, tokenizer,
            system=system, max_tokens=max_tokens,
            sampler=sampler, processors=processors, stream=stream,
        )


if __name__ == "__main__":
    main()
