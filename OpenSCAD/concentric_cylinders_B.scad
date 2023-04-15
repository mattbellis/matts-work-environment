// Units are mm

$fn = 200; // number of facets to approximate the circle

height = 90;
outer_radius = 60;

module disc(r, h, hole_r) {
    difference() {
       cylinder(r = r, h = h); // subtract a smaller cylinder to create a disc
            translate([0,0,10]) cylinder(r = hole_r, h = h+1);
    }
}

// example usage
disc(outer_radius, height, 50); // create a disc with a radius of 10, height of 2, and a hole in the center with a radius of 5

disc(40,height,30);
disc(20,height,10);

//cylinder(r=3,h=height);
//cylinder(r=3,h=height);