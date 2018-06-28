var t = 0;

var width = 710;
var height = 400;

function setup() {
  // put setup code here
    createCanvas(710, 600, WEBGL);
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
    
    t = 0;
}

function draw() {
  // put drawing code here

    // Directional light
    var dirX = (mouseX / width - 0.5) *2;
    var dirY = (mouseY / height - 0.5) *(-2);
    directionalLight(250, 250, 250, dirX, dirY, 0.25);
    ambientMaterial(250);

    background(100);
    noStroke();

    //noFill(); // This makes it into a wire frame
    stroke(255);
    push();
    translate(width*sin(t*0.01), 100, -400 + 150*cos(t*0.02));
    sphere(120);
    pop();

    t = t+1;

}
