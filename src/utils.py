import pickle

def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()