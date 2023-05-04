// Units are mm

$fn = 200; // number of facets to approximate the circle

height = 40;
outer_radius = 30;

module disc(r, h, hole_r) {
    difference() {
       cylinder(r = r, h = h); // subtract a smaller cylinder to create a disc
            translate([0,0,10]) cylinder(r = hole_r, h = h+1);
    }
}

// example usage
disc(outer_radius, height, 25); 
disc(20,height,15);
disc(10,height,5);

//cylinder(r=3,h=height);
//cylinder(r=3,h=height);


module filler(r, h, inner_r) {
    difference() {
       cylinder(r = r, h = h); // subtract a smaller cylinder to create a disc
            translate([0,0,10]) cylinder(r = inner_r, h = h+1);
    }
}

translate([50,0,0]) filler(14.5,height,10.5);
translate([100,0,0]) filler(24.5,height,20.5);
