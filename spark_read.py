# spark-submit --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 spark_read.py data/KKI.nel
import argparse

from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import split


parser = argparse.ArgumentParser(description='Perform graph classification.')
parser.add_argument('filename', type=str, help='File containing the labeled graphs')
args = parser.parse_args()

fname = args.filename


spark = SparkSession.builder.appName('classification').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')

# Read graphs from file
# raw_text = spark.read.text(fname)
# raw_text = raw_text.select(split(raw_text.value, ' '))
# print(raw_text.show(5))
graphs = []; vertices = []; edges = []
with open(fname, 'r') as f:
    for line in f:
        line = line[:-1]

        if line == '':
            vertices = spark.createDataFrame(vertices, ['id', 'label'])
            edges = spark.createDataFrame(edges, ['src', 'dst', 'label'])
            graph = GraphFrame(vertices, edges)
            graphs.append((graph, name, value))
            vertices = []; edges = []
        else:
            line = line.split(' ')

            if line[0] == 'n':
                vertices.append(line[1:])
            elif line[0] == 'e':
                edges.append(line[1:])
            elif line[0] == 'g':
                name = line[1]
            elif line[0] == 'x':
                value = float(line[1])

for g, n, v, in graphs[:3]:
    print(f'Name: {n}\nValue: {v}')
    g.outDegrees.show()

# Classify
# asdf
