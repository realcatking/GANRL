import numpy as np


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def read_w2v_embeddings(filepath):
    with open(filepath,'r') as f:
        items = f.readline().strip().split()
        num_node = int(items[0])
        embedding_size  = int(items[1])
        lines = f.readlines()
        embedding_matrix = np.random.rand(num_node, embedding_size)
        for line in lines:
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix

def write_embeddings_to_file(embedding_matrix, file_path):
    """write embeddings to files"""
    n_node = embedding_matrix.shape[0]
    embedding_size = embedding_matrix.shape[1]
    index = np.array(range(n_node)).reshape(-1, 1)
    embedding_matrix = np.hstack([index, embedding_matrix])
    embedding_list = embedding_matrix.tolist()
    embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                     for emb in embedding_list]
    with open(file_path, "w+") as f:
        lines = [str(n_node) + "\t" + str(embedding_size) + "\n"] + embedding_str
        f.writelines(lines)


def read_embeddings_from_file(file_path):
    with open(file_path,'r') as f:
        items = f.readline().strip().split()
        num_node = int(items[0])
        embedding_size  = int(items[1])
        lines = f.readlines()
        embedding_matrix = np.random.rand(num_node, embedding_size)
        for line in lines:
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix


def read_embeddings_from_d2v(model, n_node):
    embedding_matrix = np.array([model.docvecs[str(n)] for n in range(n_node)])
    return embedding_matrix
