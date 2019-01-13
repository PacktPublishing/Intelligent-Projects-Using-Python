import sys
sys.path.append('/home/santanu/ML_DS_Catalog-/Machine Translation/')
from MachineTranslation import MachineTranslation
import pickle

def inference(model_path):
    with open(model_path,"rb") as f:
        obj = pickle.load(f)
