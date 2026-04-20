{
  description = "scLDM.CD4 — Pixi-based dev shell on NixOS (pixi manages Python/CUDA/torch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pixi
          ];

          shellHook = ''
            echo "scLDM.CD4 dev shell — run 'pixi install' to set up the env, then 'pixi shell' (or a 'pixi run <task>')."
          '';

          # Use system NVIDIA driver libraries to avoid version mismatch
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
        };
      });
}
