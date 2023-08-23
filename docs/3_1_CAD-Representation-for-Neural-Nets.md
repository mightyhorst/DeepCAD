## 3.1. CAD Representation for Neural Networks
The CAD model offers two levels of representation. At the user-interaction level, a CAD model is described as a sequence of operations that the user performs (in CAD software) to create a solid shape—for example, a user may sketch a closed curve profile on a 2D plane, and then extrude it into a 3D solid shape, which is further processed by other operations such as a boolean union with another 
already created solid shape (see Fig. 2). We refer to such a
specification as a CAD command sequence.

Behind the command sequence is the CAD model’s kernel
representation, widely known as the boundary representation (or B-rep) [45, 46]. Provided a command sequence, its
B-rep is automatically computed (often through the industry
standard library Parasolid). It consists of topological components (i.e., vertices, parametric edges and faces) and the connections between them to form a solid shape.

In this work, we aim for a generative model of CAD command sequences, not B-reps. This is because the B-rep is an abstraction from the command sequence: a command
sequence can be easily converted into a B-rep, but the converse is hard, as different command sequences may result in the same B-rep. Moreover, a command sequence is humaninterpretable; it can be readily edited (e.g., by importing them into CAD tools such as AutoCAD and Onshape), allowing them to be used in various downstream applications.

### 3.1.1 Specification of CAD Commands
Full-fledged CAD tools support a rich set of commands,
although in practice only a small fraction of them are commonly used. Here, we consider a subset of the commands that are of frequent use (see Table 1). These commands fall
into two categories, namely sketch and extrusion. While conceptually simple, they are sufficiently expressive to generate a wide variety of shapes, as has been demonstrated in [48].

**CAD commands and their parameters**
> Table 1. CAD commands and their parameters. `<SOL>` indicates the start of a loop; `<EOS>` indicates the end of the whole sequence.

| Commands | Parameters |
| --- | --- |
| `<SOL>`  | ∅ | 
| L (Line) | x, y : line end-point | 
| A (Arc) |     x, y : arc end-point  |
|  |     α : sweep angle |
|  |     f : counter-clockwise flag |
| R (Circle) | x, y : center |
|  | r : radius |
| E (Extrude) |  θ, φ, γ : sketch plane orientation |
|  |  px, py, pz : sketch plane origin |
|  |  s : scale of associated sketch profile |
|  |  e1, e2 : extrude distances toward both sides |
|  |  b : boolean type, u : extrude type |
| `<EOS>` | ∅ |

> Figure 2. A CAD model example specified by the commands in Table 1. (Top) the CAD model’s construction sequence, annotated with the command types. 
> (Bottom) the command sequence description of the model. Parameter normalization and quantization are not shown in this case. In “Sketch 1”, `L2-A3-L4-L5` forms a loop (in blue) and C7 forms another loop (in green), and the two loops bounds a sketch profile (in gray).

**Parametrized command sequence:**
| index | Commands | Parameters |
| --- | --- | --- |
| 1 | `<SOL>`  | ∅ |
| 2 |  L  | (2,0)  |
| 3 |  A  |  (2,2,π, 1) |
| 4 |  L  | (0, 2)  |
| 5 |  L  | (0, 0)  |
| 6 | `<SOL>`  | ∅ |
| 7 |  R  | (2,1,0.5)  |
| 8 |  E  | (0,0,0,-2,-1,0,3, 1, 0, 'New body', 'One-sided')  |
| 9 | `<SOL>`  | ∅ |
| 10 |  R  | (0,0,1.125)  |
| 11 |  E  | (0,0,0,-2,-1,0, 2.25, 2, 0, 'Join', 'One-sided')  |
| 12 |  `<EOS>`  | ∅ |

#### 3.1.1.a. Sketch. 
Sketch commands are used to specify closed curves on a 2D plane in 3D space. In CAD terminology,
each closed curve is referred as a loop, and one or more loops form a closed region called a profile (see “Sketch 1” in Fig. 2). In our representation, a profile is described by a list of loops on its boundary; a loop always starts with an indicator command hSOLi followed by a series of curve commands `Ci`

We list all the curves on the loop in counterclockwise order, beginning with the curve whose starting point is at the most bottom-left; and the loops in a profile are sorted according to the bottom-left corners of their bounding boxes. 

> Figure 2 illustrates two sketch profiles.

In practice, we consider three kinds of curve commands that are the most widely used: draw a line, an arc, and a circle. While other curve commands can be easily added (see Sec. 5), statistics from our large-scale real-world dataset (described in Sec. 3.3) show that these three types of commands constitute 92% of the cases.

Each curve command `Ci` is described by its curve type `ti ∈ { <SOL>, L, A, R}` and its parameters listed in Table 1.

Curve parameters specify the curve’s 2D location in the sketch plane’s local frame of reference, whose own position and orientation in 3D will be described shortly in the associated extrusion command. Since the curves in each loop are concatenated one after another, for the sake of compactness
we exclude the curve’s starting position from its parameter
list; each curve always starts from the ending point of its predecessor in the loop. The first curve always starts from the origin of the sketch plane, and the world-space coordinate
of the origin is specified in the extrusion command.

In short, a sketch profile S is described by a list of loops `S = [Q1, . . . , QN ]`, where each loop Qi consists of a series of curves starting from the indicator command hSOLi (i.e., `Qi = [ <SOL>, C1, . . . , Cni]`), and each curve command `Cj = (tj , pj )` specifies the curve type ti and its shape parameters `pj` (see Fig. 2).


#### 3.1.1.b. Extrusion. 
The extrusion command serves two purposes:
1. It extrudes a sketch profile from a 2D plane into a 3D body, and the extrusion type can be either one-sided, symmetric, or two-sided with respect to the profile’s sketch plane. 
2. The command also specifies (through the parameter b in Table 1)
how to merge the newly extruded 3D body with the previously created shape by one of the boolean operations: either creating a new body, or joining, cutting or intersecting with
the existing body.

The extruded profile—which consists of one or more curve commands—is always referred to the one described immediately before the extrusion command. The extrusion command therefore needs to define the 3D orientation of that profile’s sketch plane and its 2D local frame of reference.

This is defined by a rotational matrix, determined by `(θ, γ, φ)` parameters in Table 1. This matrix is to align the world frame of reference to the plane’s local frame of reference, and to align z-axis to the plane’s normal direction. In addition, the command parameters include a scale factor s of the
extruded profile; the rationale behind this scale factor will be discussed in Sec. 3.1.2.

With these commands, we describe a CAD model `M` as a sequence of curve commands interleaved with extrusion commands (see Fig. 2). In other words, `M` is a command sequence `M = [C1, . . . , CNc]`, where each Ci has the form `(ti , pi)` specifying the command type `ti` and parameters `pi`.

