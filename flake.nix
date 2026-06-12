{
  description = "この環境はLFM2.5 8b a1b を動作させるための環境です。量子化KVキャッシュやツールコールをOpen AI互換サーバーでも利用可能にするために構築しました";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # uv.lock を単一の真実源として Nix ビルドする (uv2nix)。
    # → 開発用 .venv (uv sync) と、サーバの Nix ビルド (packages.default) が
    #   どちらも同じ uv.lock 由来になり、依存の二重管理/版ズレが起きない。
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      # pyproject.toml + uv.lock を読み込む (システム非依存)。
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # wheel 優先。mlx / mlx-metal は PyPI の macOS arm64 wheel
      # (cp314) をそのまま使い、ソースからの Metal ビルドを避ける。
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };
    in
    # Apple Silicon (Metal) 専用。mlx-metal は aarch64-darwin wheel のみ。
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # pyproject の requires-python (>=3.14) に合わせる。
        python = pkgs.python314;

        # mlx の core.so は @rpath/libmlx.dylib を必要とするが、その実体は別 wheel
        # (mlx-metal) に同梱されている。uv の .venv では両 wheel が同じ mlx/ に
        # 展開されるので解決できるが、uv2nix では wheel ごとに store path が分かれ
        # @loader_path/lib から見えなくなる。mlx-metal の mlx/lib/* を mlx 側の
        # mlx/lib/ へ symlink して橋渡しする (バイナリ非改変なので再署名も不要)。
        pyprojectOverrides = final: prev: {
          mlx = prev.mlx.overrideAttrs (old: {
            postInstall = (old.postInstall or "") + ''
              mkdir -p "$out/${python.sitePackages}/mlx/lib"
              ln -sfn "${final.mlx-metal}/${python.sitePackages}/mlx/lib/"* \
                "$out/${python.sitePackages}/mlx/lib/"
            '';
          });
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );

        # uv.lock の default 依存群 + 本プロジェクト本体を含む venv。
        # pyproject の [project.scripts] により bin/lfm2-serve・bin/lfm2-run が入る。
        venv = pythonSet.mkVirtualEnv "lfm2-agent-env" workspace.deps.default;
      in
      {
        # OpenAI 互換サーバ (lfm2-serve) と対話 CLI (lfm2-run) を含む venv。
        # 他マシンでも `nix run github:tori3-po4/LFM2.5_for_MLX` だけで動く。
        packages.default = venv;
        packages.lfm2-agent = venv;

        apps.default = {
          type = "app";
          program = "${venv}/bin/lfm2-serve";
        };
        apps.serve = {
          type = "app";
          program = "${venv}/bin/lfm2-serve";
        };
        apps.run = {
          type = "app";
          program = "${venv}/bin/lfm2-run";
        };

        devShells.default = pkgs.mkShell {
          # 開発(PyCharm 静的解析)用の Python 本体は従来どおり uv に管理させる
          # (Nix store の Python は PyCharm と相性が悪いため)。依存元は同じ uv.lock。
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
