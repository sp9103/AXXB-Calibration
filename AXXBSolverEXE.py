import SolverAXXB
import numpy as np

def start_steup():
    prompt = '>'

    print("Hand eye calibration AX=XB solver")
    print("To get started, give a path of dataset :")
    datasetpath = input(prompt)
    print("next, give a path of create result :")
    outputpath = input(prompt)

    X, error = SolverAXXB.SolveAXXBFromDataSet(datasetpath)
    print("Final Average Error : %fmm" % error)
    outputpath = outputpath + "\Te.csv"
    np.savetxt(outputpath, X, delimiter=',')

    print("Result is saved in %s, press any key to exit" % outputpath)
    input(prompt)

start_steup()