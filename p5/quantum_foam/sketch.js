//var ps = new ParticleSystem();

var mic;

var t = 0.0;

///////////////////////////////////////////////////////////////////////////////
// Info for particles
///////////////////////////////////////////////////////////////////////////////
//var x = [1000][3];
//var p = [1000][3];

var quote_timer = 0;
var quote_num = 0;

var background_color = 0;





///////////////////////////////////////////////////////////////////////////////
function setup() {
    // put setup code here
    createCanvas(windowWidth, windowHeight);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
    /////////////////////////////////////////////////////////////////////////////
    // For ParticleSystem
    /////////////////////////////////////////////////////////////////////////////
    colorMode(RGB, 255, 255, 255, 100);
    print("setup: "+windowWidth/2+ " " + windowHeight/2+ " " + 0);
    ps = new ParticleSystem(20, createVector(windowWidth/2,windowHeight/2,0.0));

    smooth();
    
    frameRate(30);

    mic = new p5.AudioIn();
    mic.start();
}

///////////////////////////////////////////////////////////////////////////////
function draw() {

    /////////////////////////////////////////////////////////////////////////////
    // Get the amplitude of input
    /////////////////////////////////////////////////////////////////////////////
    var micLevel = mic.getLevel();
    //print("tot micLevel: "+micLevel);
    //
    //if (micLevel>0.002) {
        //print("micLevel: "+micLevel);
    //}

    // put drawing code here
    //print("background_color: "+background_color);
    background(background_color);

    stroke(255);
    fill(255);
    // For debugging
    //ellipse(width/2, constrain(height-micLevel*height*5, 0, height), 10, 10);

    //////////////////////////////////////////
    // Get the Particle going
    //////////////////////////////////////////

    //print("size: "+ps.size());
    ps.run();

    // If things are loud then make new particles
    if (micLevel>0.015)
    {
        //var nparticles=(micLevel/50);
        var num_new= int(micLevel*500);
        //print("num_new: "+num_new);
        //println("size: "+ps.size());
        if (num_new <= 10)
            num_new=10;

        for (var i=0;i<num_new;i++)
        {
            var pflag=1;
            if (micLevel>0.008)
            {
                if (random(1)<0.1)
                    pflag=2;
            }
            //print("HITHERE");
            if (ps.size()<200)
                ps.addParticlePair(random(windowWidth),random(windowHeight),pflag);
            //ps.addParticle(random(screen_width),random(screen_height),1);
        }
    }



    //////////////////////////////////////////

    if (quote_timer==0) {
        quote = qft_quotes();
        //print("here!");
    }

    fill(255, 255, 255,200-quote_timer);
    stroke(200-quote_timer);
    text(quote, 15, 30,600,400);
    textSize(24);

    quote_timer += 1;

    if (quote_timer>=200) {
        quote_timer=0;
    }



    t += 1.0;
}

///////////////////////////////////////////////////////////////////////////////
window.onresize = function() {
    canvas.size(windowWidth, windowHeight);
};

///////////////////////////////////////////////////////////////////////////////
// Quotes
///////////////////////////////////////////////////////////////////////////////
function qft_quotes()
{
    //print("Quotes!");
    var s = "";
    var r = -1;
    while (r==quote_num || r==-1)
    {
        r = ceil(random(4));
        //print("r: "+r);
    }

    if (r==1) {
        s = "For those who are not shocked when they first come across quantum theory cannot possibly have understood it.\n- Niels Bohr";
    } else if (r==2) {
        s = "I think I can safely say that nobody understands quantum mechanics.\n-Richard Feynman";
    } else if (r==3) {
        s = "I do not like it, and I am sorry I ever had anything to do with it.\n-Erwin Schrodinger";
    } else if (r==4) {
        s = "What nature demands from us is not a quantum theory or a wave theory; rather, nature demands from us a synthesis of these two views which thus far has exceeded the mental powers of physicists.\n-Albert Einstein";
    }

    quote_num=r;
    return s;

}

///////////////////////////////////////////////////////////////////////////////
// A simple Particle class
///////////////////////////////////////////////////////////////////////////////
//var Particle = function(l, v, a, energy, xpflag) 
var Particle = function(l, v, a, energy, xpflag) {
    //float r;
    //float timer;
    //int R;
    //int G;
    //int B;
    //int transparency;
    //int pflag;

    // Another constructor (the one we are using here)
    //this.l = createVector(0.0, 0.0, 0.0);
    //this.v = createVector(0.0, 0.0, 0.0);
    //this.a = createVector(0.0, 0.0, 0.0);
    this.acc = a.copy();
    //acc = new PVector(0,0.0,0);
    this.vel = v.copy();
    this.loc = l.copy();
    this.r = energy;
    this.timer = 20-energy;
    this.pflag = xpflag;
    if (this.pflag==1) {
        this.R=0; this.G=191; this.B=255;
    } else if (this.pflag==-1) {
        this.R=255; this.G=191; this.B=0;
    }else if (this.pflag==2) {
        this.R=0; this.G=191; this.B=255;
        this.timer=40;
    } else if (this.pflag==-2) {
        this.R=255; this.G=191; this.B=0;
        this.timer=200;
    }
}

// Another constructor (the one we are using here)
/*
var Particle = function(l,red,green,blue) {
    var xdir=random(-2,2);
    var ydir=random(-2,2);
    this.acc = createVector(-0.05*xdir,-0.05*ydir,0);
    this.vel = createVector(xdir,ydir,0);
    this.loc = l.copy();
    //print("loc: "+this.loc);
    //print("vel: "+this.vel);
    //print("acc: "+this.acc);
    this.r = 30.0;
    this.timer = 100.0;
    this.R=red;
    this.G=green;
    this.B=blue;
    //print("blue: "+this.B);
}
*/

// Method to update location
Particle.prototype.update = function() {
    //print("UPDATE: "+this.loc+ " " +this.vel+" "+this.acc);
    this.vel.add(this.acc);
    this.loc.add(this.vel);
    //print("POSTDATE: "+this.loc+ " " +this.vel+" "+this.acc);
    //print("a: "+this.timer);
    this.timer -= 1.0;
}

// Method to display
Particle.prototype.render = function() {
    //stroke(255,timer);
    //fill(100,timer);
    stroke(this.R,this.G,this.B,40);
    fill(this.R,this.G,this.B,20);
    if (abs(this.pflag)==2)
    {
        fill(this.R,this.G,this.B,100);
    }
    ellipseMode(CENTER);
    //print("loc.x: ",this.loc.x);
    ellipse(this.loc.x,this.loc.y,this.r,this.r);
    push();
    stroke(255);
    pop();
    //displayVector(vel,loc.x,loc.y,10);
}

Particle.prototype.run = function() {
    //print("-------------");
    this.update();
    this.render();
}

// Is the particle still useful?
Particle.prototype.dead = function() {
    //print("DEAD TIMER: "+this.timer);
    isDead = this.timer <=0.0 || this.loc.x<0 || this.loc.x>windowWidth || this.loc.y<0 || this.loc.y>windowHeight;
    if (isDead) {
        return Boolean(1);
    } else {
        return Boolean(0);
    }
}

Particle.prototype.displayVector = function(v, x, y, scayl) {
    pushMatrix();
    stroke(255);
    /*
       float arrowsize = 4;
    // Translate to location to render vector
    translate(x,y);
    stroke(255);
    // Call vector heading function to get direction (note that pointing up is a heading of 0) and rotate
    rotate(v.heading2D());
    // Calculate length of vector & scale it to be bigger or smaller if necessary
    float len = v.mag()*scayl;
    // Draw three lines to make an arrow (draw pointing up since we've rotate to the proper direction)
    line(0,0,len,0);
    line(len,0,len-arrowsize,+arrowsize/2);
    line(len,0,len-arrowsize,-arrowsize/2);
     */
    popMatrix();
} 



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// A class to describe a group of Particles
// An ArrayList is used to manage the list of Particles 


//  var particles;    // An arraylist for all the particles
//  var origin;        // An origin point for where particles are born

var ParticleSystem = function(num, v) {
    this.particles = [];              // Initialize the arraylist
    //print("HERE v     : "+v);
    this.origin = v.copy();                        // Store the origin point
    //print("HERE origin: "+this.origin);
    for (var i = 0; i < num; i++) {
        //print("Adding: "+i);
        //this.particles.push(new Particle(this.origin,0,191,255));    // Add "num" amount of particles to the arraylist
        this.addParticlePair(random(windowWidth),random(windowHeight),1);
        //print("Added!");
    }
}

ParticleSystem.prototype.size = function() {
    return this.particles.length;
}

ParticleSystem.prototype.run = function() {
    // Cycle through the ArrayList backwards b/c we are deleting
    //print("++++++++++++++++++++");
    //print(this.particles.length);
    for (var i = this.particles.length-1; i >= 0; i--) {
        var p = this.particles[i];
        //print("============");
        p.run();
        //print("=================================>");
        if (p.dead()) {
            //print("REMOVING:");
            this.particles.splice(i,1);
        }
        if (this.particles.length<50)
            this.addParticlePair(random(windowWidth),random(windowHeight),1);
    }
}

//ParticleSystem.prototype.addParticle = function() {
    //this.particles.add(new Particle(origin,0,191,255));
    //this.particles.add(new Particle(origin,255,191,0));
//}

//ParticleSystem.prototype.addParticle = function(x, y,R,G, B) {
    //this.particles.push(new Particle(new PVector(x,y),R,G,B));
    //this.particles.push(new Particle(new PVector(x,y),B,G,R));
    ////print("here");
//}

ParticleSystem.prototype.addParticlePair = function(x, y, pflag) {
    var range=1.0;
    if (abs(pflag)==1)
    {
        range=2;
    }
    else if (abs(pflag)==2)
    {
        range=4;
    }
    //var px=random(-range,range);
    //var py=random(-range,range);
    var px=randomGaussian(0,range);
    var py=randomGaussian(0,range);

    //var px=0.0;
    //var py=5.0;
    var pmag=1.0*sqrt(px*px+py*py);
    var angle=atan2(py,px);
    //print("angle: "+angle);

    //print("new angle: "+cos(angle+1.57));
    //print("new angle: "+sin(angle+1.57));
    var px0=px+pmag*cos(angle+1.57);
    var py0=py+pmag*sin(angle+1.57);
    var ax0=-0.05*pmag*cos(angle+1.57);
    var ay0=-0.05*pmag*sin(angle+1.57);

    var px1=px-pmag*cos(angle+1.57);
    var py1=py-pmag*sin(angle+1.57);
    var ax1=0.10*pmag*cos(angle+1.57);
    var ay1=0.10*pmag*sin(angle+1.57);

    var energy=40.0*sqrt(ax1*ax1+ay1*ay1);
    //print("energy: "+energy);

    if (pflag==2)
    {
        ax0=ay0=ax1=ay1=0;
        energy = energy/3.0;
    }

    this.particles.push(new Particle(createVector(x,y), createVector(px0,py0), createVector(ax0,ay0), energy, pflag));
    this.particles.push(new Particle(createVector(x,y), createVector(px1,py1), createVector(ax1,ay1), energy, -pflag));
    //print("here");
}

//ParticleSystem.prototype.addParticle = function(p) {
    //this.particles.push(p);
//}

// A method to test if the particle system still has particles
ParticleSystem.prototype.dead = function() {
    if (this.particles.isEmpty()) {
        return true;
    } else {
        return false;
    }
}


