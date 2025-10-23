declare module "plotly.js-dist-min" {
  type PlotlyType = typeof import("plotly.js");

  const Plotly: PlotlyType;
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  type PlotlyType = typeof import("plotly.js");
  type PlotlyComponentType = typeof import("react-plotly.js");

  export default function createPlotlyComponent(
    plotly: PlotlyType,
  ): PlotlyComponentType;
}
