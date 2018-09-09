import numpy as np
from scipy.stats import skew
from pyquaternion import Quaternion
import SolverAXXB

def CreateAlignAtoB(A, B):
    A = identifyMarker(A)
    B = identifyMarker(B)

    A = pointstoMat(A)
    B = pointstoMat(B)

    return np.matmul(A, np.linalg.inv(B)), A, B

def toQuaternion(pitch, roll, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return w, x, y, z

def daniilidis(A, B):
    shape = A.shape
    n = int(shape[1] / 4)
    T = np.zeros((6*n, 8))

    for i in range(n):
        A1 = A[:, 4*i:4*(i+1)]
        B1 = B[:, 4*i:4*(i+1)]
        print(np.linalg.det(A1))
        print(np.linalg.det(B1))
        a = Quaternion(matrix=A1)
        b = Quaternion(matrix=B1)

        up_mat = np.concatenate((a[1:3, 0] - b[1:3, 0], skew(a[1:3, 0] + b[1:3, 1]), np.zeros((3, 4))), axis=1)
        low_mat = np.concatenate(
            (a[1:3, 1] - b[1:3, 1], skew(a[1:3, 1] + b[1, 3:1]), a[1:3, 0] - b[1:3, 0], skew(a[1:3, 0] + b[1:3, 0])))
        T[6*i:6*(i+1), :] = np.concatenate((up_mat, low_mat))

    u, s, v = np.linalg.svd(T)
    u1 = v[0:3, 6]
    v1 = v[4:7, 6]
    u2 = v[0:3, 7]
    v2 = v[4:7, 7]

    a = np.matmul(np.transpose(u1), v1)
    b = np.matmul(np.transpose(u1), v2) + np.matmul(np.transpose(u2), v1)
    c = np.matmul(np.transpose(u2), v2)

    s1 = (-b + np.sqrt(np.power(b, 2) - 4 * np.matmul(a, c))) / 2 / a
    s2 = (-b - np.sqrt(np.power(b, 2) - 4 * np.matmul(a, c))) / 2 / a

    s = np.concatenate((s1, s2))
    max_smat = np.matmul(np.power(s, 2), np.matmul(np.transpose(u1), u2)) + 2 * np.matmul(s, np.matmul(np.transpose(u1),
                                                                                                       u2)) + np.transpose(
        u2) * u2
    val = np.amax(max_smat)
    idx = np.argmax(max_smat)
    s = s[idx]
    L2 = np.sqrt(1/val)
    L1 = s * L2

    q = L1 * v[:, 6] + L2 * v[:, 7]
    X = Quaternion(q).transformation_matrix

    return X

def RandomTGen():
    RandT = np.zeros((4, 4))

    randx = np.random.rand(3)
    randy = np.random.rand(3)
    randx /= np.linalg.norm(randx)
    randy /= np.linalg.norm(randy)
    randz = np.cross(randx, randy)
    randy = np.cross(randx, randz)

    randy /= np.linalg.norm(randy)
    randz /= np.linalg.norm(randz)

    print(np.dot(randx, randy))
    print(np.dot(randy, randz))
    print(np.dot(randz, randx))

    t = np.random.rand(3)
    RandT[0:3, 0] = randx
    RandT[0:3, 1] = randz
    RandT[0:3, 2] = randy
    RandT[0:3, 3] = t
    RandT[3, 3] = 1

    calcMaxRotVecLen(RandT)

    return RandT

def identifyMarker(markerlist):
    sorted_list = []

    maxdist = 0.0
    mindist = 9999999.9
    maxidx = []
    minidx = []

    for i in range(len(markerlist)):
        idx1 = i
        idx2 = (i + 1) % 3

        marker1 = markerlist[idx1]
        marker2 = markerlist[idx2]

        length = np.linalg.norm(marker1 - marker2)

        if length > maxdist:
            maxdist = length
            if len(maxidx) == 0:
                maxidx.append(idx1)
                maxidx.append(idx2)
            else:
                maxidx[0] = idx1
                maxidx[1] = idx2

        if length < mindist:
            mindist = length
            if len(minidx) == 0:
                minidx.append(idx1)
                minidx.append(idx2)
            else:
                minidx[0] = idx1
                minidx[1] = idx2

    for i in range(2):
        for j in range(2):
            if maxidx[i] == minidx[j]:
                sorted_list.append(markerlist[maxidx[i]])
                sorted_list.append(markerlist[maxidx[(i + 1) % 2]])
                sorted_list.append(markerlist[minidx[(j + 1) % 2]])

    print("max dist : %f" % maxdist)
    print("min dist : %f" % mindist)
    return sorted_list

def pointstoMat(pointlist):
    X = np.zeros((4, 4))

    for i in range(len(pointlist)):
        point = np.array(pointlist[i])
        X[0:3, i] = point

    axis1 = pointlist[0] - pointlist[2]
    axis2 = pointlist[1] - pointlist[2]
    cross = np.cross(axis1, axis2)
    cross /= np.linalg.norm(cross)
    X[0:3, 3] = cross + pointlist[2]

    X[3, :] = 1.0

    return X

def measureDiff(AX, XB):
    diff = AX - XB
    return np.linalg.norm(diff)

def IsRotNormalized(X):
    Rx0 = X[:, 0]
    Rx1 = X[:, 1]
    Rx2 = X[:, 2]

    norm1 = np.linalg.norm(Rx0)
    norm2 = np.linalg.norm(Rx1)
    norm3 = np.linalg.norm(Rx2)

    angle1 = np.dot(Rx0, Rx1)
    angle2 = np.dot(Rx1, Rx2)
    angle3 = np.dot(Rx2, Rx0)

    print("RotNorm")
    print(norm1)
    print(norm2)
    print(norm3)
    print("Angle")
    print(angle1)
    print(angle2)
    print(angle3)

    return max([norm1, norm2, norm3])

def calcMaxRotVecLen(X):
    print("determinat : %f" % np.linalg.det(X))
    Rx = extractR(X)

    RMax = IsRotNormalized(Rx)
    return RMax

def printError(A, B):
    diff = A - B
    mat_sum = np.sum(np.matmul(diff, np.transpose(diff)))
    diff = np.sqrt(mat_sum)
    print("Error = %f" % diff)

def RandGenTest():
    X = RandomTGen()
    A_set = []
    B_set = []
    TotalA = RandomTGen()
    TotalB = np.matmul(np.matmul(np.linalg.inv(X), TotalA), X)
    A_set.append(TotalA)
    B_set.append(TotalB)
    for i in range(9):
        A = RandomTGen()
        B = np.matmul(np.matmul(np.linalg.inv(X), A), X)
        noise = np.random.normal(0, 0.1, size=(4, 4))
        B += noise
        AX = np.matmul(A, X)
        XB = np.matmul(X, B)
        print(np.linalg.norm(AX - XB))

        A_set.append(A)
        B_set.append(B)
        TotalA = np.concatenate((TotalA, A), axis=1)
        TotalB = np.concatenate((TotalB, B), axis=1)

    # X_LSC = daniilidis(TotalA, TotalB)
    X_LSC = SolverAXXB.LeastSquareAXXB(A_set, B_set)
    printError(X_LSC, X)

    for i in range(len(A_set)):
        X_calc = SolverAXXB.CalcAXXB(A_set[i], B_set[i], A_set[(i + 1) % len(A_set)], B_set[(i + 1) % len(A_set)])
        printError(X_calc, X)

def main():
    """
    point_init1 = [np.array([91.06972, -63.8444977, 560.858948]),
                   np.array([29.9324741, 55.2437477, 488.348175]),
                   np.array([-15.2011309, 27.69368, 479.593018])]
    point_step1 = [np.array([48.200428, -63.2141, 542.5844]),
                   np.array([63.5268478, 75.96184, 482.743469]),
                   np.array([10.3362284, 80.5515747, 477.897461])]
    TInit1 = np.array([0.50661969184875488, 0.85258293151855469, 0.12821389734745026, 459.28863525390625,
                       -0.43619132041931152, 0.1251857727766037, 0.89110356569290161, 295.204833984375,
                       0.74368917942047119, -0.50737637281417847, 0.43531087040901184, 415.14920043945312,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr1 = np.array([0.33690634369850159, 0.68499112129211426, 0.6459730863571167, 471.34432983398437,
                    -0.814968466758728, -0.13142247498035431, 0.56440633535385132, 323.3056640625,
                    0.47150871157646179, -0.71659976243972778, 0.51396912336349487, 420.730224609375,
                    0, 0, 0, 1]).reshape((4, 4))
    A1 = np.matmul(np.linalg.inv(TInit1), Tr1)

    point_init2 = [np.array([21.93578, -90.40951, 599.1209]),
                   np.array([26.38111, 48.6108551, 537.2459]),
                   np.array([-25.0260315, 44.5601578, 522.6269])]
    point_step2 = [np.array([-49.5645027, -99.20879, 604.045654]),
                   np.array([-41.7079773, 40.79832, 544.8035]),
                   np.array([-91.09982, 35.8372574, 524.553955])]
    TInit2 = np.array([0.28852856159210205, 0.80455255508422852, 0.51908230781555176, 384.24478149414062,
                       -0.70422953367233276, -0.1889842301607132, 0.68435788154602051, 332.07266235351562,
                       0.64870023727416992, -0.56300991773605347, 0.51206237077713013, 452.18386840820312,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr2 = np.array([0.21746976673603058, 0.83383691310882568, 0.50736862421035767, 379.14517211914062,
                    -0.65809118747711182, -0.2586330771446228, 0.70712441205978394, 328.00360107421875,
                    0.72084873914718628, -0.48767298460006714, 0.49249580502510071, 471.75579833984375,
                    0, 0, 0, 1]).reshape((4, 4))
    A2 = np.matmul(np.linalg.inv(TInit2), Tr2)

    point_init3 = [np.array([-63.100708, -118.123924, 509.206116]),
                   np.array([-66.6241455, 37.1597443, 462.608826]),
                   np.array([-16.5070667, 22.7952328, 475.095062])]
    point_step3 = [np.array([91.20978, -57.6626854, 597.800964]),
                   np.array([125.266975, 77.31433, 536.1191]),
                   np.array([73.74775, 91.99223, 537.920959])]
    TInit3 = np.array([0.21096973121166229, 0.60499429702758789, 0.76777189970016479, 484.78274536132812,
                       -0.77744603157043457, -0.372251957654953, 0.50695770978927612, 324.6578369140625,
                       0.59251111745834351, -0.703853964805603, 0.39181652665138245, 437.65896606445313,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr3 = np.array([0.3237481415271759, 0.57127928733825684, 0.75420624017715454, 482.97146606445312,
                    -0.89180552959442139, -0.082000702619552612, 0.44492560625076294, 289.77716064453125,
                    0.31602224707603455, -0.81664913892745972, 0.48292246460914612, 474.86083984375,
                    0, 0, 0, 1]).reshape((4, 4))
    A3 = np.matmul(np.linalg.inv(TInit3), Tr3)

    point_init4 = [np.array([-55.57776, -48.851738, 607.7701]),
                   np.array([-4.316303, 84.33742, 554.602356]),
                   np.array([-54.5252228, 101.754738, 547.595032])]
    point_step4 = [np.array([-74.96915, -26.8147774, 682.6206]),
                   np.array([-16.7939758, 98.08627, 617.7476]),
                   np.array([-65.86214, 118.821838, 611.7956])]
    TInit4 = np.array([0.1943543404340744, 0.59997034072875977, 0.77605539560317993, 496.32644653320312,
                       -0.87469422817230225, -0.25209617614746094, 0.4139535129070282, 266.63677978515625,
                       0.44400042295455933, -0.75926482677459717, 0.47579461336135864, 499.35549926757812,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr4 = np.array([0.13979873061180115, 0.60835909843444824, 0.78125256299972534, 464.41586303710937,
                    -0.92196398973464966, -0.20780983567237854, 0.32679882645606995, 201.5938720703125,
                    0.36116299033164978, -0.76597273349761963, 0.53183358907699585, 534.25469970703125,
                    0, 0, 0, 1]).reshape((4, 4))
    A4 = np.matmul(np.linalg.inv(TInit4), Tr4)

    point_init5 = [np.array([8.893441, -90.72155, 550.531433]),
                   np.array([11.1930513, 66.03247, 509.2107]),
                   np.array([59.27496, 43.9770432, 500.528625])]
    point_step5 = [np.array([3.97495937, -129.338074, 554.537842]),
                   np.array([23.7159424, 15.4163465, 511.709381]),
                   np.array([-28.686655, 20.3712769, 501.641])]
    TInit5 = np.array([0.33017891645431519, 0.41726228594779968, 0.84668409824371338, 476.88861083984375,
                       -0.91911894083023071, -0.062138255685567856, 0.38904905319213867, 302.91848754882812,
                       0.21494698524475098, -0.90665924549102783, 0.36299696564674377, 393.07052612304687,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr5 = np.array([0.36532396078109741, 0.69391930103302, 0.62049531936645508, 364.55902099609375,
                    -0.7200615406036377, -0.21179251372814178, 0.660798966884613, 385.37203979492187,
                    0.58995741605758667, -0.68820047378540039, 0.42229172587394714, 417.15103149414062,
                    0, 0, 0, 1]).reshape((4, 4))
    A5 = np.matmul(np.linalg.inv(TInit5), Tr5)

    point_init6 = [np.array([0.5355112, -144.733215, 559.934631]),
                   np.array([-41.3958855, 2.29159784, 505.94577]),
                   np.array([11.1576557, 0.8008199, 516.4024])]
    point_step6 = [np.array([2.30023837, -90.59772, 542.0796]),
                   np.array([-39.7096062, 56.3965454, 488.037231]),
                   np.array([12.8412085, 54.942482, 498.525574])]
    TInit6 = np.array([0.36539950966835022, 0.69386559724807739, 0.62051081657409668, 329.99996948242187,
                       -0.72002321481704712, -0.2118004709482193, 0.66083812713623047, 350,
                       0.58995735645294189, -0.68825215101242065, 0.42220756411552429, 417.11099243164062,
                       0, 0, 0, 1]).reshape((4, 4))
    Tr6 = np.array([0.36539939045906067, 0.69386559724807739, 0.62051087617874146, 370,
                    -0.72002321481704712, -0.21180066466331482, 0.66083812713623047, 390,
                    0.58995747566223145, -0.68825209140777588, 0.4222075343132019, 417.11099243164062,
                    0, 0, 0, 1]).reshape((4, 4))
    A6 = np.matmul(np.linalg.inv(TInit6), Tr6)

    E1, setInit1, setstep1 = CreateAlignAtoB(point_init1, point_step1)
    E2, setInit2, setstep2 = CreateAlignAtoB(point_init2, point_step2)
    E3, setInit3, setstep3 = CreateAlignAtoB(point_init3, point_step3)
    E4, setInit4, setstep4 = CreateAlignAtoB(point_init4, point_step4)
    E5, setInit5, setstep5 = CreateAlignAtoB(point_init5, point_step5)
    E6, setInit6, setstep6 = CreateAlignAtoB(point_init6, point_step6)

    X12 = SolverAXXB.CalcAXXB(A1, E1, A2, E2)
    X23 = SolverAXXB.CalcAXXB(A2, E2, A3, E3)
    X34 = SolverAXXB.CalcAXXB(A3, E3, A4, E4)
    X24 = SolverAXXB.CalcAXXB(A2, E2, A4, E4)

    A_set = [A1, A2, A3, A4, A5]
    B_set = [E1, E2, E3, E4, E5]
    X_LS = SolverAXXB.LeastSquareAXXB(A_set, B_set)

    T1X = np.matmul(A1, X12)
    XB1 = np.matmul(X12, E1)
    T2X = np.matmul(A2, X12)
    XB2 = np.matmul(X12, E2)

    printError(T1X, XB1)
    printError(T2X, XB2)

    TransformDiff = A1 - np.matmul(X12, np.matmul(E1, np.linalg.inv(X12)))
    #Align = X^inv * T1^inv * T2 * X
    VerifyResult(X_LS, TInit1, Tr1, setInit1, setstep1)
    VerifyResult(X_LS, TInit2, Tr2, setInit2, setstep2)
    VerifyResult(X_LS, TInit3, Tr3, setInit3, setstep3)
    VerifyResult(X_LS, TInit4, Tr4, setInit4, setstep4)
    VerifyResult(X_LS, TInit5, Tr5, setInit5, setstep5)
    VerifyResult(X_LS, TInit6, Tr6, setInit6, setstep6)

    np.savetxt("Te.csv", X_LS, delimiter=',')
    """
    SolverAXXB.SolveAXXBFromDataSet("./DataSets")

    return

if __name__ == "__main__":
    main()