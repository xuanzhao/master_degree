from io import BytesIO as StringIO
from IPython.display import Image
from sklearn import tree
import pydotplus


def draw_DecTree(DecTree, feat_names=None, cla_names=None):
	# from sklearn.externals.six import StringIO  
	# import pydotplus 
	# dot_data = StringIO() 
	# tree.export_graphviz(DecTre, out_file=dot_data) 
	# graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
	# graph.write_pdf("iris.pdf") 

	dot_data = StringIO()
	# tree.export_graphviz(DecTree, out_file=dot_data)
	tree.export_graphviz(DecTree, out_file=dot_data,  
	                     feature_names=feat_names,  
	                     class_names=cla_names, 
	                     node_ids=True, 
	                     filled=True, rounded=True,  
	                     special_characters=True)  
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	graph.write_pdf('dot_data.pdf') 

	# return Image(graph.create_png())