"""Create (or update) the W&B report with the parallel-coordinates chart of the sweep.

    pip install wandb-workspaces
    python scripts/wandb_report.py [--entity ENTITY] [--project PROJECT] [--url EXISTING_REPORT]

The chart has the same axes as the sample in the assignment:
activation bits, weight bits, compression ratio, model size (MB), quantized accuracy.
Only the compression runs (job types "ptq" and "qat") are shown.
"""
import argparse

import wandb
import wandb_workspaces.reports.v2 as wr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=None, help="defaults to your W&B default entity")
    p.add_argument("--project", default="cs6886w-a2-mobilenetv2-compression")
    p.add_argument("--url", help="update this existing report instead of creating a new one")
    args = p.parse_args()
    entity = args.entity or wandb.Api().default_entity

    runset = wr.Runset(entity=entity, project=args.project, name="Compression runs",
                       filters="JobType in ['ptq', 'qat']")
    columns = [
        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("activation_quant_bits"),
                                         display_name="activation_quant_bits"),
        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("weight_quant_bits"),
                                         display_name="weight_quant_bits"),
        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("compression_ratio"),
                                         display_name="compression_ratio"),
        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("model_size_mb"),
                                         display_name="model_size_mb"),
        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("quantized_acc"),
                                         display_name="quantized_acc"),
    ]
    parallel = wr.ParallelCoordinatesPlot(title="MobileNet-v2 compression sweep", columns=columns,
                                          layout=wr.Layout(x=0, y=0, w=24, h=11))
    scatter = wr.ScatterPlot(title="Accuracy vs. compression ratio",
                             x=wr.SummaryMetric("compression_ratio"),
                             y=wr.SummaryMetric("quantized_acc"),
                             layout=wr.Layout(x=0, y=11, w=12, h=8))
    scatter_act = wr.ScatterPlot(title="Accuracy vs. activation compression ratio",
                                 x=wr.SummaryMetric("act_compression_ratio"),
                                 y=wr.SummaryMetric("quantized_acc"),
                                 layout=wr.Layout(x=12, y=11, w=12, h=8))
    curves = wr.LinePlot(title="test_acc during quantization-aware fine-tuning", x="epoch",
                         y=["test_acc"], max_runs_to_show=50,
                         layout=wr.Layout(x=0, y=19, w=24, h=8))

    report = (wr.Report.from_url(args.url) if args.url
              else wr.Report(entity=entity, project=args.project))
    fields = dict(
        title="CS6886W A2: MobileNet-v2 compression sweep",
        description="Parallel coordinates of every compression configuration: bit-widths, "
                    "in-memory compression ratio, model size and test accuracy.",
        blocks=[
            wr.H1("MobileNet-v2 on CIFAR-10: compression sweep"),
            wr.P("Each line is one compression run. weight_quant_bits is the average number of "
                 "bits per weight actually stored (mixed-precision runs have fractional values); "
                 "compression_ratio and model_size_mb are in-memory numbers that include every "
                 "overhead; quantized_acc is the test accuracy of the packed model."),
            wr.PanelGrid(runsets=[runset], panels=[parallel, scatter, scatter_act, curves]),
        ],
        width="fluid",
    )
    for k, v in fields.items():
        setattr(report, k, v)
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
