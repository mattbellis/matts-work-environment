// Units are mm

$fn = 200; // number of facets to approximate the circle

height = 40;
r0 = 20;
r1 = r0*sqrt(2);
r2 = r0*sqrt(3);


module disc(r, h, hole_r) {
    difference() {
       cylinder(r = r, h = h); // subtract a smaller cylinder to create a disc
            translate([0,0,10]) cylinder(r = hole_r, h = h+1);
    }
}

// example usage
disc(r2+2, height, r2); 
disc(r1+2,height,r1);
disc(r0+2,height,r0);

//cylinder(r=3,h=height);
//cylinder(r=3,h=height);