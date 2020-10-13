def visualize_pygal_scatter(embedded):
    """
    Simple SVG scatter plot with Pygal (requires installation).
    :param embedded: A numpy array with two columns.
    """
    import pygal  # pip install pygal
    from IPython.display import SVG, display  # Needed for Jupyter Notebooks, not sure how it works elsewhere
    from pygal.style import CleanStyle
    xy_chart = pygal.XY(stroke=False, style=CleanStyle)
    xy_chart.title = 'TSNE'
    xy_chart.add('TSNE', embedded)
    display(SVG(xy_chart.render(disable_xml_declaration=True)))