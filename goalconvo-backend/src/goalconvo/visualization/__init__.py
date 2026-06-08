"""Publication-quality figures (matplotlib, IEEE-style, PNG + PDF)."""

from .publication_figures import (
    apply_ieee_matplotlib_style,
    export_ablation_figure_bundle,
    export_evaluation_figure_bundle,
    plot_coherence_comparison_overlay,
    save_figure_png_pdf,
)

__all__ = [
    "apply_ieee_matplotlib_style",
    "export_ablation_figure_bundle",
    "export_evaluation_figure_bundle",
    "plot_coherence_comparison_overlay",
    "save_figure_png_pdf",
]
