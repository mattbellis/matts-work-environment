var t = 0;

var width = 710;
var height = 400;

var nspheres = 10;
var x0 = [];
var y0 = [];

function setup() {
  // put setup code here
    createCanvas(710, 600, WEBGL);
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
    
    t = 0;

    for(var i=0;i<nspheres;i++) {
        x0.push(random(100,500));
        y0.push(random(100,500));
    }
}

function draw() {
  // put drawing code here
    background(100);
    noStroke();

    push();
    translate(-100,-100,-400);
    sphere(100);
    pop();

    // Directional light
    //var dirX = (mouseX / width - 0.5) *2;
    //var dirY = (mouseY / height - 0.5) *(-2);
    //directionalLight(250, 250, 250, dirX, dirY, 0.25);
    //ambientMaterial(250);

    //noFill(); // This makes it into a wire frame
    stroke(255);

    for(var i=0;i<nspheres;i++)
    {
        push();
        translate(x0[i] + 140*sin(t*0.01), y0[i]+130*cos(t*0.02), -400)
        sphere(20);
        pop();
    }

    t = t+1;

}
