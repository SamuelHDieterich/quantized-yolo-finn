{
  config,
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
    directory = "./src";
    uv = {
      enable = true;
      sync.enable = true;
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
