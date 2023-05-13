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
//disc(outer_radius, height, 25); 
disc(outer_radius,height,20*sqrt(2));
disc(22,height,20);

//cylinder(r=3,h=height);
//cylinder(r=3,h=height);


module filler(r, h, inner_r) {
    difference() {
       cylinder(r = r, h = h-10); // subtract a smaller cylinder to create a disc with no bottom
            translate([0,0,0]) cylinder(r = inner_r, h = h+1);
    }
}

translate([60,0,0]) filler(19.5,height,0);
translate([120,0,0]) filler(outer_radius-2.5,height, 22 + .5);
