# spark-submit --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 graph_processing.py
from graphframes import GraphFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Sample').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Sample social media
users = spark.createDataFrame(
    [
        ('1', 'Dog', 'Theodora'),
        ('2', 'Giraffe', 'Borys'),
        ('3', 'Dog', 'Motke'),
        ('4', 'Cat', 'Jöns'),
        ('5', 'Fox', 'Aoede'),
        ('6', 'Fox', ' Sara'),
        ('7', 'Cat', 'Brynja'),
        ('8', 'Cat', 'Mārtiņš'),
        ('9', 'Cat', 'Nuur'),
    ],
    ['id', 'type', 'name'],
)
messages = spark.createDataFrame(
    [
        ('2', '6'),
        ('7', '1'),
        ('9', '3'),
        ('9', '3'),
        ('6', '3'),
        ('9', '3'),
        ('5', '8'),
        ('1', '4'),
        ('8', '2'),
        ('8', '1'),
        ('5', '6'),
        ('2', '3'),
        ('2', '7'),
        ('8', '4'),
    ],
    ['src', 'dst'],
)

g = GraphFrame(users, messages)

# Graph info
print('Graph Outgoing and Incoming Edges')
g.inDegrees.show()
g.outDegrees.show()

# Motif finding
print('Unreciprocated Messages')
g.find('(a)-[]->(b); !(b)-[]->(a)').show()

# Page rank
print('Page Rank')
pagerank = g.pageRank(tol=.01)
pagerank.vertices.show()
pagerank.edges.show()
