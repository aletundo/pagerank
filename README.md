# PageRank
A naive implementation of the PageRank algorithm based on the power method.

## Requirements
All the required Python packages are listed in the `requirements.txt` file.
For instance, you can install them using [pip](https://pypi.org/project/pip/) by running the command below:

```shell
pip install -r requirements.txt
```

## How to run
The file `main.py` contains the algorithm implementation and some utility methods.
The script arguments can be shown by running `python3 main.py --help`. For the sake of simplicity they are reported below:

| Argument | Description | Default value |
|---|---|---|
| -h, --help | Show the help message and exit |  |
| --nodes | The number of nodes of the generated graph | 25 |
| --edge_probability | The edge probability of the generated graph | 0.125 |
| --seed | The random seed used during the graph generation and for plots layout | 100 |
| --dumping_factor | The dumping factor used by the algorithm | 0.85 |
| --iterations | The number of iterations of the algorithm | 100 |

For instance, you can run `python3 main.py --nodes 50` to generate a graph with 50 nodes while using the default values
for all the other arguments.

Some informative messages providing details about the current step will appear in the console during the script execution.
Moreover, the naive algorithm implementation is compared with the [NetworkX](https://networkx.org/documentation/stable/index.html) implementation.
The results of both the algorithms are reported at the end of the execution along with a plot (only when nodes are <= 50).

## License
This project is licensed under the GPL v3.0 license. See the [LICENSE](LICENSE) file for details.

