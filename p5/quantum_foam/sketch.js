var ps = new ParticleSystem();

var mic;

var t = 0.0;

///////////////////////////////////////////////////////////////////////////////
// Info for particles
///////////////////////////////////////////////////////////////////////////////
var x = [1000][3];
var p = [1000][3];

var quote_timer = 0;
var quote_num = 0;




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
        ps = new ParticleSystem(0, new PVector(width/2,height/2,0));
          smooth();

    mic = new p5.AudioIn();
  mic.start();
}

///////////////////////////////////////////////////////////////////////////////
function draw() {

     /////////////////////////////////////////////////////////////////////////////
      // Get the amplitude of input
      /////////////////////////////////////////////////////////////////////////////
      var tot = mic.getLevel();
        if (tot>0.002) {
            print("tot: "+tot);
      }

  // put drawing code here
    background(0);

    micLevel = mic.getLevel();
    stroke(255);
    fill(255);
   ellipse(width/2, constrain(height-micLevel*height*5, 0, height), 10, 10);

    if (quote_timer==0) {
        quote = qft_quotes();
        print("here!");
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
    print("Quotes!");
    var s = "";
    var r = -1;
    while (r==quote_num || r==-1)
    {
        r = ceil(random(4));
        print("r: "+r);
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
var Particle = function(l,red,green,blue) {
    var xdir=random(-2,2);
    var ydir=random(-2,2);
    this.acc = createVector(-0.05*xdir,-0.05*ydir,0);
    //acc = new PVector(0,0.0,0);
    this.vel = createVector(xdir,ydir,0);
    this.loc = l.copy();
    this.r = 30.0;
    this.timer = 100.0;
    this.R=red;
    this.G=green;
    this.B=blue;
    print(B);
  }

  Particle.prototype.run = function() {
    update();
    render();
  }

  // Method to update location
  Particle.prototype.update = function() {
    this.vel.add(this.acc);
    this.loc.add(this.vel);
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
    ellipse(this.loc.x,this.loc.y,this.r,this.r);
    pushMatrix();
    stroke(255);
    popMatrix();
    //displayVector(vel,loc.x,loc.y,10);
  }
  
  // Is the particle still useful?
  var dead = function() {
    if (this.timer <= 0.0) {
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

class ParticleSystem {

  var particles;    // An arraylist for all the particles
  var origin;        // An origin point for where particles are born

  ParticleSystem(var num, PVector v) {
    particles = new ArrayList();              // Initialize the arraylist
    origin = v.get();                        // Store the origin point
    for (var i = 0; i < num; i++) {
      particles.add(new Particle(origin,0,191,255));    // Add "num" amount of particles to the arraylist
    }
  }

  var size() {
    return particles.size();
  }

  void run() {
    // Cycle through the ArrayList backwards b/c we are deleting
    for (var i = particles.size()-1; i >= 0; i--) {
      Particle p = (Particle) particles.get(i);
      p.run();
      if (p.dead()) {
        particles.remove(i);
      }
    }
  }

  void addParticle() {
    particles.add(new Particle(origin,0,191,255));
    particles.add(new Particle(origin,255,191,0));
  }
  
  void addParticle(var x, var y,var R,var G, var B) {
    particles.add(new Particle(new PVector(x,y),R,G,B));
    particles.add(new Particle(new PVector(x,y),B,G,R));
    //print("here");
  }

  void addParticle(var x, var y, var pflag) {
    var range=1.0;
    if (abs(pflag)==1)
    {
      range=10;
    }
    else if (abs(pflag)==2)
    {
      range=30;
    }
    var px=random(-range,range);
    var py=random(-range,range);
    //var px=0.0;
    //var py=5.0;
    var pmag=1.0*sqrt(px*px+py*py);
    var angle=atan2(py,px);
    print("angle: "+angle);

    print("new angle: "+cos(angle+1.57));
    print("new angle: "+sin(angle+1.57));
    var px0=px+pmag*cos(angle+1.57);
    var py0=py+pmag*sin(angle+1.57);
    var ax0=-0.10*pmag*cos(angle+1.57);
    var ay0=-0.10*pmag*sin(angle+1.57);

    var px1=px-pmag*cos(angle+1.57);
    var py1=py-pmag*sin(angle+1.57);
    var ax1=0.10*pmag*cos(angle+1.57);
    var ay1=0.10*pmag*sin(angle+1.57);

    var energy=40.0*sqrt(ax1*ax1+ay1*ay1);
    print("energy: "+energy);

    if (pflag==2)
    {
      ax0=ay0=ax1=ay1=0;
      energy = energy/3.0;
    }

    particles.add(new Particle(new PVector(x,y), new PVector(px0,py0), new PVector(ax0,ay0), energy, pflag));
    particles.add(new Particle(new PVector(x,y), new PVector(px1,py1), new PVector(ax1,ay1), energy, -pflag));
    //print("here");
  }

  void addParticle(Particle p) {
    particles.add(p);
  }

  // A method to test if the particle system still has particles
  boolean dead() {
    if (particles.isEmpty()) {
      return true;
    } else {
      return false;
    }
  }

}

