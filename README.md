![Preliminary Example](./webs/docs/prelim_example_phi5.png){#id .class width=30 height=20px}

The Horsefly problem is a generalization of the well-known Euclidean
Traveling Salesman Problem. In the most basic version of the Horsefly
problem, which we call the "classic" horsefly, we are given a set of
sites in the Euclidean plane, the initial position of a truck (horse)
with a drone (fly) mounted on top, and the speed of the drone
\$\\varphi\$. [^1] [^2]

The goal is to compute a tour for both the truck and the drone to
deliver package to sites as quickly as possible. For delivery, a drone
must pick up a package from the truck, fly to the site and come back to
the truck to pick up the next package for delivery to another site. Both
the truck and drone must coordinate their motions to minimize the time
it takes for all the sites to get their packages, i.e. minimize the
makespan of the delivery process.

This suite of programs implement several experimental heuristics, to
approximately solve this NP-hard problem and some of its variations.

Please see `horseflies.pdf` in the `tex` folder for a full description
of the installation instructions, possible variations on Classic
Horsefly and the correponding programs for solving them written in a
[literate](http://www.literateprogramming.com/knuthweb.pdf) format. [^3]

[^1]: The speed of the truck is always assumed to be 1.

[^2]: \$\\varphi\$ is also called the speed ratio.

[^3]: The specific software that I used to write these literate programs
    was [Nuweb](http://nuweb.sourceforge.net/nuweb.pdf)
