from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Sphere_3
from CGAL.CGAL_Kernel import ON_BOUNDED_SIDE
from CGAL.CGAL_Kernel import do_intersect
from CGAL.CGAL_Kernel import intersection


from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Kernel import Segment_2

p1=Point_2(3.14,0)
p2=Point_2(3.14,0.1)

p3=Point_2(0,0)
p4=Point_2(1,0)

print "Comparing points", p1 == p2
print p1.x(), p1.y()

print do_intersect(Segment_2(p1,p2), Segment_2(p3,p4))
print intersection(Segment_2(p1,p2), Segment_2(p3,p4))


from CGAL import CGAL_Kernel
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Kernel import Triangle_2
t1=Triangle_2(Point_2(0,0),Point_2(1,0),Point_2(0,1))
t2=Triangle_2(Point_2(1,1),Point_2(1,0),Point_2(0,1))
object = CGAL_Kernel.intersection(t1,t2)
assert object.is_Segment_2()
print object.get_Segment_2()
