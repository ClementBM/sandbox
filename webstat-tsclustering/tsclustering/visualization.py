# création d'un graph des datasets de webstat
import networkx as nx

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

graph = nx.DiGraph()

for dataset_name in dataset_names:
    if dataset_name in series_dict:
        series = series_dict[dataset_name]
    else:
        series = get_series_by_dataset_name(dataset_name)
        series_dict[dataset_name] = series

    for serie in series:
        serie_key = serie["seriesKey"]
        serie_dataset = serie["dataset"]
        serie_title = serie["title"]

        graph.add_edges_from([(serie_dataset, serie_title if serie_title is not None else serie_key)])

graph_renderer = from_networkx(graph, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)

plot = Plot(width=1000, height=1000, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
plot.renderers.append(graph_renderer)

output_file("interactive_graphs.html")

# show(plot)

# plt.figure(figsize=(12,12)) 
# nx.draw(graph, with_labels=True, node_size=60,font_size=8)
