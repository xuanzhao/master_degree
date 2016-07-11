plt.plot([2.99,2.99], [-1.7,1.7], color='blue', linewidth=1.0, linestyle="-")
plt.plot([1.2,1.2], [-1.7,1.7], color='blue', linewidth=0.8, linestyle="-")
plt.plot([5.65,5.65], [-1.7,1.7], color='blue', linewidth=0.8, linestyle="-")
plt.plot([0,1.2], [0.38,0.38], color='red', linewidth=0.6, linestyle="-")
plt.plot([-0.5,1.2], [0.38,0.38], color='red', linewidth=0.6, linestyle="-")
plt.plot([1.2,2.99], [0.03,0.03], color='red', linewidth=0.6, linestyle="-")
plt.plot([2.99,5.65], [-0.12,-0.12], color='red', linewidth=0.6, linestyle="-")
plt.plot([2.99,5.65], [0.53,0.53], color='red', linewidth=0.6, linestyle="-")
plt.plot([0.37,0.37], [-1.5,0.38], color='blue', linewidth=0.4, linestyle="-")
plt.plot([0.37,0.37], [0.38,1.5], color='blue', linewidth=0.4, linestyle="-")
plt.plot([0.37,0.37], [0.38,1.7], color='blue', linewidth=0.4, linestyle="-")
plt.plot([2.89,2.89], [-1.7,0.03], color='blue', linewidth=0.4, linestyle="-")
plt.plot([3.45,3.45], [-1.5,-0.12], color='blue', linewidth=0.4, linestyle="-")
plt.plot([2.99,5.65], [-0.48,-0.48], color='red', linewidth=0.4, linestyle="-")

plt.xticks(np.linspace(0, 7, 15, endpoint=True))
plt.yticks(np.linspace(-1.5, 1.5, 9, endpoint=True))


RMat(clf)
array([[[ 0.14396752,  0.09313125],
        [ 0.04265158,  0.2341086 ]],

       [[ 0.84518619,  0.22525511],
        [ 0.77053279,  0.23457581]],

       [[ 1.99574121,  0.46433913],
        [ 0.84309485,  0.36386319]],

       [[ 4.60422781,  0.6047853 ],
        [-0.82756663,  0.30389907]],

       [[ 6.39590107,  0.27911976],
        [ 0.0828144 ,  0.25668035]]])

splitIndex [0]<2.996638 
    splitIndex [0]<1.206384 
        splitIndex [1]<0.385154 
            splitIndex [0]<0.373538 
                leaf node:  [[-0.07906004  2.54498897]]
                leaf node:  0
            splitIndex [0]<0.373538 
                leaf node:  1
                leaf node:  [[-0.07906004  2.54498897]]
        splitIndex [1]<0.030807 
            leaf node:  0
            splitIndex [0]<2.898392 
                leaf node:  [[-0.07906004  2.54498897]]
                leaf node:  1
    splitIndex [0]<5.651739 
        splitIndex [1]<-0.122368 
            splitIndex [0]<3.456388 
                leaf node:  0
                leaf node:  [[-0.07906004  2.54498897]]
            leaf node:  1
        splitIndex [1]<0.530558 
            splitIndex [1]<-0.486141 
                leaf node:  0
                leaf node:  [[-0.07906004  2.54498897]]
            leaf node:  1








plt.xticks(np.linspace(0, 7, 15, endpoint=True))
plt.yticks(np.linspace(-1.5, 1.5, 9, endpoint=True))


plt.plot([2.99,2.99], [-1.7,1.7], color='blue', linewidth=1.0, linestyle="-")
plt.plot([2.7,2.7], [-1.7,1.7], color='blue', linewidth=0.8, linestyle="-")
plt.plot([-.5,2.7], [1.23,1.23], color='red', linewidth=0.6, linestyle="-")
plt.plot([2.7,2.9], [-0.01,-0.01], color='red', linewidth=0.6, linestyle="-")
plt.plot([1.2,1.2], [-1.7,1.23], color='blue', linewidth=0.4, linestyle="-")
plt.plot([2.99,7.5], [-1.27,-1.27], color='red', linewidth=0.8, linestyle="-")
plt.plot([2.99,7.5], [0.7,0.7], color='red', linewidth=0.6, linestyle="-")
plt.plot([6.9,6.9], [-1.27,0.7], color='blue', linewidth=0.4, linestyle="-")

myFore.trees[0].tree.getTreeStruc()
    splitIndex [0]<2.996638 
        splitIndex [0]<2.715025 
            splitIndex [1]<1.238367 
                splitIndex [0]<1.206384 
                    leaf node:  [[-0.05966642  2.54390819]]
                    leaf node:  [[-0.05966642  2.54390819]]
                leaf node:  1
            splitIndex [1]<-0.010027 
                leaf node:  0
                leaf node:  [[-0.05966642  2.54390819]]
        splitIndex [1]<-1.271662 
            leaf node:  0
            splitIndex [1]<0.703506 
                splitIndex [0]<6.920313 
                    leaf node:  [[-0.05966642  2.54390819]]
                    leaf node:  0
                leaf node:  1

RMat(label and clf) 
array([[[ 0.54302883,  0.34388149],
        [ 0.44070716,  0.40820959]],

       [[ 2.00098614,  0.41362832],
        [ 0.8256669 ,  0.29126009]],

       [[ 1.63093446,  0.22975672],
        [ 1.32104224,  0.12387357]],

       [[ 2.92208182,  0.01538047],
        [-0.07515812,  0.09866428]],

       [[ 2.89022532,  0.07298946],
        [ 0.27200313,  0.21070963]],

       [[ 4.98401599,  0.17319131],
        [-1.31046339,  0.03656352]],

       [[ 5.17908873,  1.01227775],
        [-0.92005099,  0.52505714]],

       [[ 6.93387409,  0.02369293],
        [ 0.35556195,  0.01859097]],

       [[ 6.84009032,  0.06635618],
        [ 0.77738196,  0.05582279]]])
