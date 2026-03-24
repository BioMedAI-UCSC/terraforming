# Formula/tform.rb
#
# Homebrew tap formula for the tform CLI.
#
# To use this tap:
#   brew tap BioMedAI-UCSC/terraforming https://github.com/BioMedAI-UCSC/terraforming
#   brew install tform
#
# Or install directly without tapping:
#   brew install BioMedAI-UCSC/terraforming/tform

class Tform < Formula
  include Language::Python::Virtualenv

  desc     "Interactive Mars climate simulation CLI"
  homepage "https://github.com/BioMedAI-UCSC/terraforming"
  # Update url + sha256 each release.  Run: brew fetch --build-from-source tform
  url      "https://github.com/BioMedAI-UCSC/terraforming/archive/refs/tags/v0.1.0.tar.gz"
  sha256   "REPLACE_WITH_ACTUAL_SHA256"
  license  "MIT"

  # tform is a Python application — Homebrew's virtualenv helper installs it
  # in an isolated venv so it never pollutes the system Python.
  depends_on "python@3.12"

  # ── Python dependencies ────────────────────────────────────────────────────
  # Keep versions in sync with uv.lock / pyproject.toml.
  # Generate sha256 with: pip download --no-deps <pkg>==<ver>; shasum -a 256 *.whl

  resource "click" do
    url    "https://files.pythonhosted.org/packages/source/c/click/click-8.1.8.tar.gz"
    sha256 "ed53c9d8821d0d0f2bfb73f35b7f4b1d9f3e65d24d6cce2b41dd7f9d6afe0862"
  end

  resource "pyyaml" do
    url    "https://files.pythonhosted.org/packages/source/P/PyYAML/PyYAML-6.0.2.tar.gz"
    sha256 "d584d9ec91ad65861cc08d42e834324ef890a082e591037abe114850ff7bbc3e"
  end

  # NOTE: torch is large (~800 MB CPU wheel).  The formula installs the
  # CPU-only variant.  GPU support requires a manual `pip install` into the
  # Cellar venv after brew install.
  resource "torch" do
    url    "https://files.pythonhosted.org/packages/source/t/torch/torch-2.6.0.tar.gz"
    sha256 "REPLACE_WITH_ACTUAL_SHA256"
  end

  resource "matplotlib" do
    url    "https://files.pythonhosted.org/packages/source/m/matplotlib/matplotlib-3.10.1.tar.gz"
    sha256 "REPLACE_WITH_ACTUAL_SHA256"
  end

  def install
    # Build a virtualenv inside the Cellar and install all Python deps + tform.
    # Language::Python::Virtualenv handles PATH wiring automatically.
    virtualenv_install_with_resources
  end

  test do
    # Smoke-test: version flag must exit 0 and print a version string.
    assert_match "tform", shell_output("#{bin}/tform --version")
  end
end
