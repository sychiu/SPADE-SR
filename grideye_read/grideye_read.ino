#include <SparkFun_GridEYE_Arduino_Library.h>
#include <Wire.h>

float pixelValues[64];
float deviceTemp = 25.0;
long t = 0;
GridEYE grideye;

void setup() {
  // Start your preferred I2C object 
  Wire.begin();
  // Library assumes "Wire" for I2C but you can pass something else with begin() if you like
  grideye.begin();
  // Pour a bowl of serial
  Serial.begin(115200);
  //grideye.setFramerate10FPS();
  //grideye.movingAverageDisable();
}


void loop() {

  if(millis()-t>100){
    t = millis();
    deviceTemp = grideye.getDeviceTemperature();
    for(unsigned int i = 0; i < 64; i++){
      pixelValues[i] = grideye.getPixelTemperature(i);
    }
  

    for(unsigned int i = 0; i < 64; i++){
      Serial.print(pixelValues[i]);
      Serial.print(',');
    }
    Serial.println(deviceTemp);
    
  }
  
  


}
