{
  config,
  lib,
  pkgs,
  ...
}:
{
  packages = with pkgs; [
    git # Version control
    beads # Issue tracker (bd)
  ];

  languages.python = {
    enable = true;
    package = pkgs.python313;
    directory = ".";
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # opencv-python (pulled in by ultralytics) ships prebuilt binaries linked
  # against shared libs that don't exist outside FHS distros.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath (
    with pkgs;
    [
      zlib # libz.so.1
      stdenv.cc.cc.lib # libstdc++.so.6
      xorg.libxcb # libxcb.so.1
      libGL # libGL.so.1
      glib # libglib-2.0.so.0, libgthread-2.0.so.0
    ]
  );

  scripts = {
    ########
    # Data #
    ########

    download-data =
      let
        output_path = "${config.devenv.root}/data";
        folder_name = "ships-in-aerial-images";
      in
      {
        description = "Download the Ships in Aerial Images dataset from Kaggle and configure it for YOLO.";
        packages = with pkgs; [
          curl
          unzip
          gnused
        ];
        exec = ''
          # Download to a temporary location
          temp_folder=$(mktemp -d)
          curl -L -o $temp_folder/${folder_name}.zip \
            https://www.kaggle.com/api/v1/datasets/download/siddharthkumarsah/${folder_name}

          # Extract to the data directory
          unzip $temp_folder/${folder_name}.zip -d ${output_path}

          # Patch the dataset YAML to use absolute paths instead of Kaggle-specific ones.
          # Ultralytics requires resolvable paths; the Kaggle zip ships /kaggle/input/... paths.
          DATA_YAML="${output_path}/${folder_name}/data.yaml"
          sed -i \
            -e "s|train:.*|train: ${output_path}/${folder_name}/train/images|" \
            -e "s|val:.*|val: ${output_path}/${folder_name}/valid/images|" \
            -e "s|test:.*|test: ${output_path}/${folder_name}/test/images|" \
            "$DATA_YAML"

          echo "Dataset ready at: ${output_path}/${folder_name}/"
          echo "Config patched:   $DATA_YAML"

          # Clean up the archive
          rm $temp_folder/${folder_name}.zip
          rmdir $temp_folder
        '';
      };

    ###################
    # Pipeline stages #
    ###################

    # Phase 1: Standard full-precision training via Ultralytics
    train = {
      description = ''
        Train a YOLO model (full precision).
        Pass --data <dataset.yaml> and optional overrides.
      '';
      exec = "uv run python -m qyf.stages.train \"$@\"";
    };

    # Phase 2: Quantization-aware fine-tuning via Brevitas
    qat = {
      description = ''
        Fine-tune a trained YOLO model with Brevitas QAT.
        Pass --weights <best.pt> --data <dataset.yaml> and optional overrides.
      '';
      exec = "uv run python -m qyf.stages.qat \"$@\"";
    };
  };

  enterShell = ''
    # Activate the Python virtual environment
    source ${config.devenv.state}/venv/bin/activate
  '';

  # Git Hooks
  git-hooks.hooks = {
    # Python
    ruff.enable = true;
    ruff-format.enable = true;

    # Beads issue tracker integration.
    # Run as a Nix-managed hook step
    # (rather than via `bd hooks install` writing directly into .git/hooks/pre-commit)
    beads = {
      enable = true;
      name = "beads";
      entry = "${config.devenv.root}/scripts/beads-pre-commit-hook.sh";
      language = "system";
      pass_filenames = false;
      always_run = true;
    };
  };
}
