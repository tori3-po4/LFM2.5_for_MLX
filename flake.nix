{
  description = "この環境はLFM2.5 8b a1b を動作させるための環境です。量子化KVキャッシュやツールコールをOpen AI互換サーバーでも利用可能にするために構築しました";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          # Python 本体は uv に管理させ、Nix は開発ツールのみ提供する
          # （Nix store の Python は PyCharm の静的解析と相性が悪いため）
          packages = [
            pkgs.uv
            pkgs.ruff
            pkgs.pyright
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "uv sync で .venv を作成します..."
              uv sync
            fi
          '';
        };
      }
    );
}
