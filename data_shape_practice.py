from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

'''
To better understand dimensionality
Here we have a typical timestep data.
[m, t, n]
m=2, batch number. 
    The feature value will be x.1x for first example.
    The feature value will be x.2x for second example. 
t=2. timestep. we have 3 timestep
    The value pattern willbe t.mn
n=3: each timestep has 3 value.

The final feature value has t.mn pattern.

'''
data = tf.constant(
    [
        [
            [1.11,1.12,1.13],
            [2.11, 2.12,2.13],
            [3.11, 3.12, 3.13],
        ],
        [
            [1.21,1.22,1.23],
            [2.21,2.22,2.23],
            [3.21, 3.22, 3.23]
        ],
    ]
)
#(2,2,3)
print(data.shape)

#------------concate doesn't change dim or rank
print("--------concatenate at axis 0")
print(tf.concat([data[0], data[1]], 0))

print("--------concatenate at axis 1")
print(tf.concat([data[0], data[1]], 1))


#------------stack add new dimension at specified axis
print ('''-------stack at axis 0: batch level
         shape will be (m, t, n)
       ''')
mtn=tf.stack([data[0], data[1]], 0)
print(mtn)

print ('''-------stack at axis 1: timestep level 
timestep becomes first dimension
result[0] will be data for timestep 1.
result[0][0] will be data for time step 1, example 1

result will be (t, m, n)

''')
tmn=tf.stack([data[0], data[1]], 1)
print(tmn)

print ('''-------stack at axis 2: value level
 result[0] is timestep.
 result[0][0] is value for first feature 
 result[0][0][0] will be the value for first feature of first example
 result[0][0][1] will be the value for first feature of second example.
 shape will be (t, n, m)
 ''')
tnm=tf.stack([data[0], data[1]], 2)
print(tnm)

print("-------unstack at batch dimension")
print(tf.unstack(mtn, axis=0))


print("-------unstack at timestep dimension")
print(tf.unstack(mtn, axis=1))



print("-------unstack at value dimension")
print(tf.unstack(mtn, axis=2))
