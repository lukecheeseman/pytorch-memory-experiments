import plotly.graph_objects as go
import sys, os, json

# plot "time" against memory where each time stamp is a discrete event

def create_trace(log, key, name):
    scale = 1024 ** 2
    baseline = log[0][key]

    ys = []
    text = []
    prev = log[0][key] / scale
    for entry in log:
        y = entry[key] / scale
        ys.append(y)
        text.append(f'{entry["layer_type"]}:{entry["hook_type"]} {y - prev:+}')
        prev = y

    return go.Scatter(
            x = list(range(len(log))),
            y = ys,
            text = text,
            mode = 'lines',
            name = f'{name}'
        )

def plot(infiles, outfile):
    layout = go.Layout(
            title = 'memory usage increase',
            xaxis = dict(
                title = 'Events',
                showgrid = False
            ),
            yaxis = dict (
                title = 'Memory (MiB)'
            ),
            hovermode='closest'
    )
    fig = go.Figure(layout = layout)
    for infile in infiles:
        with open(infile, 'r') as jsonfile:
            logfile = json.load(jsonfile)

            log = logfile["run"]
            # trim to only "fwd" and "bwd" events
            log = list(filter(lambda entry: entry["hook_type"] != "pre", log))

            fig.add_trace(create_trace(log, "pytorch_mem_all", f"{infile}_py_mem"))
            fig.add_trace(create_trace(log, "pytorch_mem_cached", f"{infile}_py_cache"))
            fig.add_trace(create_trace(log, "nvml_mem_used", f"{infile}_vml_mem"))

            # find phase changes
            prev = ""
            for idx, entry in enumerate(log):
                curr = entry["hook_type"]
                if prev != curr and (prev == "bwd" or curr == "bwd"):
                    fig.add_vline(x = ((idx - 1) + idx) / 2, line_dash="dash")
                prev = curr

            scale = 1024 ** 2
            expected = logfile["expected"] / scale
            fig.add_trace(go.Scatter(
                x = list(range(len(log))),
                y = [ expected for _ in log ],
                text = [ 'TBA' for entry in log ],
                mode = 'lines',
                name = 'expected'
            ))

    fig.write_html(outfile)


if __name__ == "__main__":
    logfiles = sys.argv[1:]

    plot(logfiles, "out.html")