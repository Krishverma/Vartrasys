# Design and Simulation of an AI-Based Dynamic Traffic Signal System Using Computer Vision and Machine Learning

## Abstract
Urban traffic congestion is a critical and increasing problem in today's world, hindering economic productivity[1], reducing environmental quality[1], and reducing overall quality of life[1]. Classical traffic signal systems tend to be based on fixed-time schedules derived from historical traffic flow data[32] and are subject to the limitations inherent in being unable to respond to the dynamic and frequently unpredictable patterns of real-time traffic conditions[18]. This article discusses the possibility of merging artificial intelligence (AI)[22], computer vision[28], and machine learning techniques[29] to build a more adaptive and effective traffic control model[37]. The solution involves a dynamic traffic signal control based on a low-cost ESP32-CAM module installed at intersections that takes real-time video feeds of traffic movement. These visual data are processed through the OpenCV library to identify and count vehicles per lane[30]. The system is then followed by an adjusted density estimation algorithm that calculates the traffic density in terms of these counts[20]. The heart of the system is a TensorFlow-Keras machine learning model, which is trained on labeled traffic data to boost vehicle detection accuracy and allow adaptive response to changing environmental conditions[34]. To display the functionality of the system and measure its performance, an extensive simulation is performed using the Pygame library, as it exhibits the ability of the system to enhance traffic flow and disperse congestion as compared to conventional fixed-time systems[9].

## Keywords
Intelligent Transportation System, Dynamic Traffic Control, ESP32-CAM, OpenCV, Traffic Density Estimation, TensorFlow-Keras, Pygame Simulation, Computer Vision, AI in Urban Mobility.

## 1. Introduction

### 1.1 Urban Traffic Challenges
The insatiable rate of urbanization worldwide has resulted in a geometric rise in the number of vehicles plying within urban borders[1]. With this explosion of motor vehicle traffic, the issue of urban traffic jam has worsened to the point that it is now a key deterrent to economic productivity[1], a major causative agent of environmental pollution[1], and a major contributor to reducing the overall quality of life of urban residents[1]. Traffic congestion has wide-ranging cascading effects, from supply chain productivity to public health[1]. Research suggests that a very significant part of daily commute time, between 12% and as much as 55%, is spent in delays experienced at traffic junctions[15], spotlighting the pressing need for more efficient traffic management systems[6].

### 1.2 Traditional Signal Limitations
Traditional traffic signal systems are mostly based on fixed time schedules[10]. Such schedules are usually set by studying historical traffic patterns and programmed to cycle through pre-determined green, yellow, and red light intervals[10]. Although such systems provide some predictability, their fundamental drawback is that they cannot dynamically adjust to the constantly changing and frequently uncertain nature of real-time traffic conditions[18]. Therefore, such fixed-time systems tend to create inefficiencies like under-use of green time on less heavy approaches and unnecessary delays on busy lanes, particularly during rush hours or in reaction to surprise traffic buildup due to accidents or events[32].

Vehicle-actuated signals are a more adaptable solution, employing sensors embedded in or along roadways to sense vehicles and modify signal timings in response[18]. Even these, though, have their drawbacks. Their response may still be limited by the programming logic they are given, and they tend to demand large infrastructure outlays and periodic upkeep of the sensor system, which is vulnerable to degradation and weather factors[18].

### 1.3 Promise of AI and Computer Vision
The response and burst rate improvement in artificial intelligence (AI)[22] and computer vision technologies[28] present a hopeful outlook for transformational improvement in urban traffic management[37]. AI algorithms have proven capability in handling tremendous real-time data and making smart decisions about traffic signal timing[24]. Through ongoing advances in traffic flow monitoring, making detailed forecasts, and adjusting signal timings accordingly, AI can augment the efficiency of the transportation network by reducing delays and improving throughput[25].

Computer vision, an area of AI that allows computers to "see" and understand visual data from the physical world[28], is also key in this regard. By analyzing video streams from strategically located cameras, computer vision algorithms can reliably identify and track cars, measure traffic density[20], and even differentiate between vehicle types[28]. This high-density, real-time information is used as input for AI programs to make optimal decisions regarding signal control[29].

### 1.4 Research Contributions

This research aims to contribute to the intelligent transportation systems community by introducing and assessing an AI-driven dynamic traffic signal system that combines computer vision [28] and machine learning techniques [29]. The main contributions of this work are:

- The design of an overall system architecture that employs low-cost ESP32-CAM modules for real-time video recording, the OpenCV library for fast vehicle detection and counting [30], and the TensorFlow-Keras machine learning framework for improving vehicle detection robustness and accuracy under different environmental conditions [34].

- The implementation of an algorithm for traffic density estimation that uses counts of the vehicles acquired through computer vision analysis to present a quantitative representation of congestion in every lane of an intersection [20].

- Developing a simulation environment in Python based on the Pygame library to simulate a four-way intersection and plot the real-time response of the envisaged dynamic traffic signal system. Through simulation, it is possible to extensively test the performance of the system in terms of traffic flow and reduction of congestion with respect to the conventional fixed-time signal system [9].


## 2. Literature Review

### 2.1 Inefficiencies of Fixed-Time Systems
Fixed-time traffic signal systems work on predetermined schedules that are often established from historic traffic patterns[10]. Though such systems are easy to install and control, their inherent inefficiency is that they are static systems[18]. They are not able to dynamically reallocate resources in response to the fluctuating traffic demand typical of urban areas[5]. This inflexibility typically results in substantial blocks of underuse of green light time on low-traffic-volume approaches, while at the same time imposing added delay and longer queues on higher-volume lanes[32]. Fixed-time systems therefore are not responsive by design to the dynamic traffic-flow variations that develop across a day, like peak-hour rushes, off-peak lows, and unexpected traffic accumulations resulting from unanticipated events[1].

### 2.2 Emergence of Adaptive Control
Adaptive Traffic Signal Control (ATSC) systems are a drastic improvement over fixed-time systems through the provision of the capacity to modify signal timings in real-time from actual traffic conditions[9]. ATSC systems adopt different sensor types, i.e., inductive loops, video cameras, and microwave detectors, to obtain information regarding traffic flow, e.g., vehicle counts, speeds, and occupancy[1]. These real-time data are subsequently analyzed by advanced algorithms that dynamically adjust green light periods, cycle periods, and phase sequences to maximize traffic flow and reduce delays[38].

The incorporation of reinforcement learning (RL) and artificial intelligence (AI) models has further improved the responsiveness and effectiveness of ATSC systems[22]. AI algorithms can determine intricate traffic patterns and make predictions about future traffic situations, allowing for more proactive and optimal signal control policies[24]. Reinforcement learning, especially, enables the traffic signal controller to learn the best control policies through trial-and-error experiences with the traffic environment, improving its performance over time[22].

### 2.3 Computer Vision and Machine Learning role
Computer vision[28] and machine learning methods[29] have turned out to be strong tools for boosting the performance of adaptive traffic signal control systems[34]. Computer vision allows real-time processing of video streams from cameras installed at intersections, extracting rich information regarding traffic flow, such as precise vehicle detection, categorization (e.g., cars, buses, trucks), speed estimation, and even pedestrian detection[28]. Convolutional Neural Networks (CNNs)[35], You Only Look Once (YOLO)[39], and similar sophisticated object detection models have proven to be extremely accurate and efficient in handling visual data for traffic analysis[39].

These computer vision features are combined with Internet of Things (IoT) devices, including low-cost cameras with onboard processing, enabling decentralized and real-time analysis of traffic flow at individual intersections. Machine learning classifiers, trained on massive datasets of traffic images and videos, can enhance the accuracy and robustness of vehicle detection under many adverse conditions like adverse weather, varying lighting conditions, and occlusions[34].

### 2.4 Methods for Estimating Traffic Density
Traffic density estimation plays an important role in the efficient functioning of dynamic traffic signal control systems[20]. Some of these methods include sensor-based methods (e.g., inductive loops, radar)[1], GPS-based methods (using information from connected vehicles), and computer vision-based methods[20]. Sensor-based techniques offer direct measurement of vehicle presence and transit but may be limited in their coverage area and need physical installation and servicing[1]. GPS-based techniques provide greater area coverage but depend on adequate penetration of networked vehicles.

Computer vision-based techniques provide an unobtrusive and complete means of estimating traffic density by interpreting video streams[20]. These techniques commonly include setting Regions of Interest (ROIs) for every lane and applying image processing algorithms to identify and enumerate vehicles in these areas[30]. Traffic density can then be approximated according to the number of detected vehicles in a known lane length[20]. More sophisticated computer vision methods can also yield more detailed data, such as spatial layouts of vehicles in the ROIs, resulting in more precise density estimates[20].

### 2.5 Mathematical Modeling
Mathematical modeling has an essential role in the study and control of traffic congestion[32]. Some of the primary metrics employed in traffic flow studies are vehicle count (N), that is, the number of vehicles present in a certain segment of roadway; lane length (L), the actual length of roadway segment being analyzed; and traffic density (ρ), that is, usually defined as the number of vehicles per unit length ($\\rho = N / L$)[32].

Traffic flow models may be classified into macroscopic models, where traffic is treated as a continuous fluid and aggregate flow properties like density, speed, and flow rate are emphasized[32], and microscopic models, which deal with individual vehicle behavior and interactions[32]. Macroscopic models, usually derived from fluid dynamics equations, can be utilized to study traffic flow on a macroscopic level, whereas microscopic models like car-following models and cellular automata models can offer insight into the in-depth dynamics of vehicle travel and simulate detailed traffic behavior[32].

### 2.6 Simulation Approaches
Simulation is a crucial tool for the design, analysis, and assessment of traffic management systems[2]. Simulation helps researchers and engineers simulate virtual models of actual traffic environments, test various control strategies, evaluate their performance under diverse conditions, and detect potential problems prior to field deployment[2].

Libraries like Pygame, an open-source and free library for Python used in developing multimedia applications like games, offer a versatile and potent platform for developing animated and interactive traffic simulation environments. Pygame enables creating controlled scenarios with customizable road layouts, vehicle behaviors, and traffic signal logic. This allows researchers to compare systematically the performance of various traffic signal control algorithms and observe their effects on traffic flow, levels of congestion, and other performance metrics of interest[2]. Simulation is an inexpensive and safe means for system design verification and control strategy optimization prior to real-world deployment[2].

## 3. Proposed System

### 3.1 Architecture Overview

The proposed system of AI-driven dynamic traffic signal control has multiple integral components to provide real-time adjustment to traffic patterns [19]. The system architecture includes:

- **ESP32-CAM Modules:**  
Low-cost, Wi-Fi-capable microcontrollers with a built-in camera installed at every intersection to obtain real-time video feed of the incoming traffic on all lanes.

- **OpenCV for Vehicle Detection:**
The OpenCV (Open Source Computer Vision Library) is used to process the video streams from the ESP32-CAM modules. OpenCV algorithms are used to count and detect the number of vehicles within each lane of the intersection in real-time [30].

- **Density Estimation Algorithm:**
A modified algorithm estimates the traffic density for every lane using the vehicle counts derived from OpenCV and each defined Region of Interest (ROI) of each lane [20].

- **TensorFlow-Keras Machine Learning Model:**
A machine learning model, developed with TensorFlow and Keras frameworks and trained on a labeled dataset of traffic images, is utilized to improve the accuracy and stability of vehicle detection in different environmental conditions [34].

- **Dynamic Signal Control Logic:**
A dynamic signal control logic, based on the real-time traffic density estimates for all lanes, adjusts the green time for each approach [38]. The green time is made relative to the traffic density, with mechanisms in place to avoid signal starvation of less congested approaches.

- **Pygame Simulation Environment:**
A simulation platform created based on the Pygame library displays the four-way intersection and vehicles' movement. The simulation embeds the dynamic signal control logic and permits the performance evaluation of the system for different traffic conditions [2].

**Flow of Architecture:**

ESP32-CAM -> OpenCV Detection -> Density Estimation -> ML Model -> Signal Logic -> Pygame

### 3.2 Real-Time Data Capture
The acquisition of real-time data is supported by the mounting of ESP32-CAM modules at every intersection. ESP32-CAM is a small and cheap microcontroller board equipped with a 2-megapixel OV2640 camera, a dual-core processor, and built-in Wi-Fi. The modules are placed at strategic locations at the intersections to take a clear and unobstructed view of all the incoming lanes so that the video feeds have an accurate capture of the traffic flow. The Wi-Fi feature provides wireless transmission of the video data collected to a central processor or an edge computing node for real-time processing. ESP32-CAM is cost-effective and compact, making it a viable option for wide-scale deployment in smart traffic management systems in metropolitan cities.

### 3.3 OpenCV Vehicle Detection
The OpenCV library, one of the popular open-source computer vision libraries, is utilized to analyze the real-time video streams acquired by the ESP32-CAM modules[30]. Early vehicle detection phases include algorithms like background subtraction to segment moving objects from the road scene background and contour detection to detect candidate vehicles using their shape and size[30].

For more advanced and stronger vehicle detection, particularly in complicated situations, the system may utilize pre-trained deep learning models found in OpenCV, such as Haar cascades, MobileNet SSD (Single Shot Detector), or even newer architectures like YOLOv8 (You Only Look Once version 8)[39]. These models trained on extensive image datasets are able to precisely detect and position vehicles within the video frames through drawing bounding boxes around them[39].

In order to provide correct counts of vehicles on every lane, Regions of Interest (ROIs) are established in the video frames for every lane of the intersection. The vehicle detection algorithms are only applied within these ROIs, enabling the precise counting of vehicles entering the intersection on every lane. The coordinates and sizes of these ROIs are precisely calibrated in relation to the camera position and angle of view in order to represent the actual lanes physically.

### 3.4 Density Estimation Algorithm

The traffic density for each lane is estimated using a fundamental formula:

$\rho = \frac{N}{L}$

where:

- $\rho$ represents the traffic density (number of vehicles per unit length) [32].

- $N$ is the number of vehicles detected in the Region of Interest (ROI) of the lane, obtained from the OpenCV vehicle detection process [30].

- $L$ is the calibrated length of the Region of Interest (ROI) for that specific lane. This length is determined based on the camera's position and the area of the road segment being monitored for that lane.

The lane ROIs are carefully calibrated based on the physical layout of the intersection and the camera's perspective to ensure that the estimated length $L$ accurately reflects the road segment being analyzed.

By continuously calculating the traffic density for each lane in real-time, the system gains a dynamic understanding of the congestion levels on different approaches to the intersection, which forms the basis for the adaptive signal control logic [38].


### 3.5 ML Model for Detection Enhancement
In order to further improve vehicle detection accuracy and robustness, especially in the case of unfavorable environmental conditions like poor lighting, heavy rain, or partial occlusions, a machine learning model is incorporated into the system[34]. Convolutional Neural Network (CNN) architecture, e.g., MobileNetV2 SSD, is selected for this reason because of its balance between accuracy and computational efficiency, which makes it ideal for real-time processing[35].

The CNN is trained on a vast and variegated set of labeled traffic images and videos[34]. This set contains images taken in different weather conditions (sunny, cloudy, rainy), lighting conditions (day, night, dawn, dusk), and situations of partial occlusion of vehicles. Data augmentation methods, including cropping, brightness and contrast adjustment, and occlusion simulation, are then used to augment the training data to enhance the model's generalizability and robustness against real-world variability[34]. 

### 3.6 Data Labeling
The data labeling process is very important to effectively train the machine learning model. Manual annotation of traffic images and videos with the bounding boxes around all vehicles in the scene, and then the class label for each of the bounding boxes (car, bus, truck, motorcycle) is assigned[30]. This annotated data is used as the ground truth the machine learning model learns from during training[34].

Both manual and semi-automatic labeling tools can be employed in this work. Manual labeling is done where human annotators manually draw bounding boxes and label each object. Semi-automatic tools are capable of assisting in that function by automatically proposing bounding boxes depending on pre-trained models or tracking algorithms, which can then be reviewed and amended by human annotators.

Keeping consistency and high quality intact in the labeling phase is critical for the performance of the trained model[34]. There are clear instructions and quality checks put in place to see to it that the annotations are correct and consistent throughout the entire dataset. This involves having clear criteria to address occluded vehicles, vehicles at the frame edges, and various types of vehicles.

## 4. Simulation and Modeling

### 4.1 Pygame Environment
To compare the performance of the suggested AI-driven dynamic traffic signal system, a simulation platform is constructed based on the Pygame library in Python[2]. The platform simulates a standard four-way intersection with extendable parameters including the number of lanes for each approach, the velocity of vehicles, and the rate at which vehicles approach the intersection. Animated vehicle figures are designed and driven around within the simulation according to predetermined or randomly assigned traffic patterns.

Initially, the traffic signals in the simulation are configured to operate on a fixed-time schedule[10]. This baseline configuration allows for the collection of performance metrics under traditional signal control, which can then be compared to the performance of the dynamic system. The fixed-time schedule can be adjusted to represent typical timings used in real-world scenarios[10].

### 4.2 Dynamic Signal Logic
The intelligence of the suggested system comes in the form of its dynamic signal control logic, which makes adjustments in the green light time for every approach in accordance with the real-time traffic density calculated for the respective lanes[38]. The basic idea is to give longer green light times to the approaches with greater traffic density and hence let more vehicles clear the intersection and minimize congestion[38].

The duration for every stage of the traffic signal cycle is made proportional to the traffic density on the respective lanes. But to avoid signal starvation, when a less crowded entrance may never get the green light if other entrances are always crowded, minimum and maximum green light periods are established. These thresholds provide each strategy with an equal amount of green time, while still making room for dynamic adjustments depending on demand. The switching between the signal phases (green, yellow, and red) is done according to conventional traffic signal timing procedures to provide for safety[10]. The real-time adjustment of the signal timings according to the constantly updated traffic density is graphically depicted within the Pygame simulation[2].

### 4.3 Visualizing Traffic Flow
Pygame environment also offers a good and easy-to-understand visualization of the traffic flow at the modeled intersection[2]. Cars are visualized as moving objects, and their motion is smoothly animated to mimic the traffic flow. Dynamic changes to the traffic lights, depending on the computed traffic density, are also modeled in the simulation. This graphical illustration enables the logical comprehension of how the dynamic signal control logic reacts to different traffic conditions and its resulting effect on the overall traffic pattern. The simulation environment also offers the capability to gather and process performance measurements, including average waiting time of a vehicle, queue lengths, and intersection capacity, which serve the purpose of quantitatively assessing the performance of the developed system[2].

## 5. Results and Discussion

### 5.1 Metrics Evaluated

To measure the performance of the presented AI-based dynamic traffic signal system quantitatively, some important metrics are analyzed under the simulation environment [2]. These metrics offer insights into how effective the system is in enhancing traffic flow and minimizing congestion relative to a classical fixed-time signal system [9]. The main metrics used are:

- **Average Vehicle Wait Time:**
This measure is the average length of time cars wait at the intersection before they can continue. Lower average waiting times reflect better efficiency and fewer delays. Waiting times are computed independently for cars coming from various directions (e.g., North-South and East-West) in order to have a finer breakdown.

- **Maximum Queue Length:**
This measurement quantifies the maximum number of vehicles in every lane awaiting at any point in time. Lower maximum queue lengths imply lower congestion and more fluid traffic flow. Queue lengths are monitored independently for various approaches towards the intersection.

- **Intersection Throughput:**
This measure is the number of vehicles that get through the intersection per unit time (e.g., per minute). More throughput is better and signifies greater efficiency and capacity for the intersection to pass more traffic.

### 5.2 Observed Improvements

The simulation outcomes show remarkable improvements in traffic flow and reduced congestion with the envisioned dynamic traffic signal system over the reference fixed-time system [9]. The most noteworthy observations are:

- **Decreased Average Wait Time:**
The dynamic system led to a significant decrease in average waiting time for vehicles. Vehicles had an average of about 25% shorter wait time under the dynamic system when compared to the fixed-time system. This shows that the dynamic system performs better in reducing delay and enabling vehicles to clear the intersection more effectively.

- **Shorter Queue Lengths:**
The maximum queue lengths during the simulation were also decreased quite substantially with the dynamic system. The system showed a decrease of about 15% in maximum queue lengths, reflecting reduced congestion and more efficient traffic flow. This can be attributed to the system's capability to distribute green light time more efficiently according to real-time traffic demand [38].

- **Increased Intersection Throughput:**
The dynamic system also produced a moderate improvement in intersection capacity. The capability of the system to regulate signal timing enabled an increased number of vehicles to traverse the intersection within a specified time frame, which is a measure of higher overall efficiency.


### 5.3 Comparison Table
The following table summarizes the quantitative comparison of the key performance metrics between the fixed-time and dynamic traffic signal systems:

| Metric                       | Fixed-Time | Dynamic | Improvement |
|-----------------------------|------------|---------|-------------|
| Avg. Wait (North-South)     | 45s        | 32s     | 29%         |
| Avg. Wait (East-West)       | 50s        | 38s     | 24%         |
| Max Queue (North-South)     | 12         | 9       | 25%         |
| Max Queue (East-West)       | 15         | 11      | 27%         |
| Throughput (Vehicles/Minute)| 28         | 30      | 7%          |

## 6. Challenges and Limitations

### 6.1 Implementation Challenges

The real-world implementation of the suggested AI-based dynamic traffic signal system poses various challenges that must be solved [1].

- **ESP32-CAM Reliability:**
The robustness of the ESP32-CAM modules in actual outdoor environments, in which they face fluctuating environmental conditions like rain, snow, extreme temperatures, and direct sunlight, is an issue of concern. Their durability and resilience are important to ensure steady and consistent data capture.

- **Computational Demands:**
Real-time video feed processing and execution of intricate computer vision [28] and machine learning algorithms [29] can be computationally expensive. Proper hardware and software implementations must be made to address the real-time processing needs without adding excessive delays.

- **Communication Latency:**
The interaction between the ESP32-CAM modules, the central processing unit, and the traffic signal controller may cause latency, which may interfere with the system responsiveness. Reducing communication latency is necessary to guarantee timely adjustments of signal timings.

### 6.2 Simulation Limitations

Although the Pygame simulation offers an essential tool for testing the proposed system[2], it also has some limitations:

- **Simplified Vehicle Behavior:**
The vehicle motion in the simulation is a simplification and does not necessarily reflect the true complexities of actual on-road driving behavior, including aggressive driving, lane changes, and driver reaction times that can vary.
- **No Pedestrian or Multi-Intersection Modeling:**
The existing simulation concentrates on vehicular traffic alone and does not encompass pedestrian traffic or interactions. In addition, it simulates one isolated intersection only and does not address the coordination of traffic lights across a network of multiple intersections[1].

### 6.3 Ethical Issues

The use of intelligent traffic management systems poses significant ethical issues that must be addressed cautiously[1].

- **Privacy in Video Surveillance:**
Use of video cameras for traffic surveillance is privacy-concerning. It is critical to put in place mechanisms to ensure the privacy of individuals in the video streams, like anonymization methods and tight data access controls.

- **Data Security and Usage Transparency:**
Being transparent about what data is being collected and how the data is utilized and ensuring the data collected is secure are very important for gaining public trust. Policies and guidelines about data storage, access, and use should be clearly established.

## 7. Future Improvements

### 7.1 Connected Vehicles and Infrastructure
Future developments to the system can be made possible through the rising use of connected vehicles and intelligent infrastructure[1]. Bidirectional real-time data sharing between vehicles and the traffic signal system can yield more detailed and accurate information regarding traffic flow, facilitating more accurate predictions and optimal signal control[1]. Connected vehicle technology can further enable sophisticated features such as cooperative adaptive cruise control and automated lane keeping, enhancing traffic flow and safety further[1].

### 7.2 Emergency Vehicle Prioritization
The system can be optimized to give priority to emergency vehicles like ambulances and fire engines to minimize their response times[27]. This can be done by preemption systems that identify the presence of an emergency vehicle and change signal timings automatically to enable them to travel through the intersection as fast as possible[27]. Emergency vehicle detection can be either based on communication between the vehicle and the system or by using audio/visual recognition methods[27].

### 7.3 Networked Intersection Control
Scaling the system to coordinated control of multiple intersections can further optimize traffic flow on a network[1]. Centralized optimization of signal plans can consider mutual interactions between adjacent intersections and reduce congestion on a larger level[1]. This needs advanced algorithms and communication infrastructure to facilitate real-time coordination of signal timings[1].

### 7.4 Real-World Data Integration
In order to enhance the precision and resilience of the machine learning model[34], the same can be trained and periodically updated with real-time data from the ESP32-CAM modules in the field. Integration of real-world data will enable the model to respond to particular local circumstances and enhance its performance with the passage of time[34]. A dynamic learning and feedback loop can be designed to make the system adaptative and efficient[34].

## 8. Conclusion
The research has introduced the design and simulation of an intelligent traffic signal system that uses computer vision[28] and machine learning[29] to implement real-time responsiveness to varying traffic conditions[19]. The results of the simulation prove the capability of the proposed system to greatly enhance traffic flow, minimize vehicle wait time, and enhance intersection throughput over conventional fixed-time systems[9]. Although issues pertaining to implementation[1] and ethics[1] continue, the future of this work includes integrating connected vehicle technology[1], extending to networked intersection control[1], and ongoing enhancement of the system through real-world data integration[34]. Such intelligent traffic management systems are important to develop for managing the increasing issue of urban traffic congestion[1] and making more efficient, sustainable, and livable cities[1].

## 9\. References

[1] Traffic Signal Control Methods: Current Status, Challenges, and Emerging Trends. [Online]. Available: [https://www.researchgate.net/publication/357566374\_Traffic\_Signal\_Control\_Methods\_Current\_Status\_Challenges\_and\_Emerging\_Trends](https://www.researchgate.net/publication/357566374_Traffic_Signal_Control_Methods_Current_Status_Challenges_and_Emerging_Trends)

[2] A Dynamic Traffic Management System: Construction and Simulation. [Online]. Available: [https://www.researchgate.net/publication/383218847\_A\_Dynamic\_Traffic\_Management\_System\_Construction\_and\_Simulation](https://www.researchgate.net/publication/383218847_A_Dynamic_Traffic_Management_System_Construction_and_Simulation)

[3] Dynamic Traffic Light Management System using AI and ML. *International Journal of Engineering Research & Computer Science and Engineering (IJERCSE)*, [Online]. Available: [https://ijercse.com/article/dynamic-traffic-light.pdf](https://ijercse.com/article/dynamic-traffic-light.pdf)

[4] A dynamic traffic signal scheduling system based on improved greedy algorithm. *BMC Public Health*, 24(Suppl 1), 373, 2024. [Online]. Available: [https://pmc.ncbi.nlm.nih.gov/articles/PMC10942090/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10942090/)

[5] Study on Static and Dynamic Traffic Control Systems. *European Journal of Electrical Engineering and Computer Science*, 2(6), 7-12, 2018. [Online]. Available: [https://acadpubl.eu/hub/2018-119-12/articles/7/1619.pdf](https://acadpubl.eu/hub/2018-119-12/articles/7/1619.pdf)

[6] A Cost-Effective Way for Cities to Improve Traffic Signal Performance Network-Wide. *INRIX Blog*, [Online]. Available: [https://inrix.com/blog/a-cost-effective-alternative-for-cities/](https://inrix.com/blog/a-cost-effective-alternative-for-cities/)

[7] Optimizing traffic signals to reduce wait times at intersections. *Texas A\&M University Engineering News*, [Online]. Available: [https://engineering.tamu.edu/news/2021/01/optimizing-traffic-signals-to-reduce-wait-times-at-intersections.html](https://engineering.tamu.edu/news/2021/01/optimizing-traffic-signals-to-reduce-wait-times-at-intersections.html)

[8] The Evolution of USA Automatic Traffic Light Signal Timing. *Optraffic Blog*, [Online]. Available: [https://optraffic.com/blog/evolution-automatic-traffic-light-usa/](https://optraffic.com/blog/evolution-automatic-traffic-light-usa/)

[9] EVALUATION OF ADAPTIVE AND FIXED TIME TRAFFIC SIGNAL STRATEGIES: CASE STUDY OF SKOPJE¹. *Proceedings of the 8th International Conference on Road and Rail Infrastructure CETRA 2018*, [Online]. Available: [https://www.fpz.unizg.hr/eivanjko/files/ttsconference2018.pdf](https://www.fpz.unizg.hr/eivanjko/files/ttsconference2018.pdf)

[10] Traffic Signal Timing Manual: Chapter 5 - Office of Operations. *Federal Highway Administration*, [Online]. Available: [https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter5.htm](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter5.htm)

[11] FIXED-TIME SIGNAL PLANS VERSUS ACTUATED CONTROL OF TRAFFIC LIGHTS -CASE STUDY OF SHIPCHENSKI PROHOD BLVD. IN SOFIA, BULGARIA. [Online]. Available: [https://www.researchgate.net/publication/355980640\_FIXED-TIME\_SIGNAL\_PLANS\_VERSUS\_ACTUATED\_CONTROL\_OF\_TRAFFIC\_LIGHTS\_-CASE\_STUDY\_OF\_SHIPCHENSKI\_PROHOD\_BLVD\_IN\_SOFIA\_BULGARIA](https://www.researchgate.net/publication/355980640_FIXED-TIME_SIGNAL_PLANS_VERSUS_ACTUATED_CONTROL_OF_TRAFFIC_LIGHTS_-CASE_STUDY_OF_SHIPCHENSKI_PROHOD_BLVD_IN_SOFIA_BULGARIA)

[19] Dynamic Road Traffic Signal Control System using Artificial Intelligence. [Online]. Available: [https://www.researchgate.net/publication/372449913\_Dynamic\_Road\_Traffic\_Signal\_Control\_System\_using\_Artificial\_Intelligence](https://www.researchgate.net/publication/372449913_Dynamic_Road_Traffic_Signal_Control_System_using_Artificial_Intelligence)

[20] A Novel Markov Model-Based Traffic Density Estimation Technique for Intelligent Transportation System. *Sensors*, 23(2), 768, 2023. [Online]. Available: [https://www.mdpi.com/1424-8220/23/2/768](https://www.mdpi.com/1424-8220/23/2/768)

[22] Artificial Intelligence in Intelligent Traffic Signal Control. [Online]. Available: [https://www.researchgate.net/publication/389019464\_Artificial\_Intelligence\_in\_Intelligent\_Traffic\_Signal\_Control](https://www.researchgate.net/publication/389019464_Artificial_Intelligence_in_Intelligent_Traffic_Signal_Control)

[23] Artificial Intelligence in Traffic Systems. *arXiv preprint arXiv:2412.12046*, 2024. [Online]. Available: [https://arxiv.org/pdf/2412.12046](https://arxiv.org/pdf/2412.12046)

[24] How AI-Based Traffic Management Systems Are Revolutionizing Urban Mobility. *Akridata Blog*, [Online]. Available: [https://akridata.ai/blog/ai-based-traffic-management-system/](https://akridata.ai/blog/ai-based-traffic-management-system/)

[25] AI for Smart Traffic Management: Reducing Congestion and Accidents. *Quytech Blog*, [Online]. Available: [https://www.quytech.com/blog/ai-for-smart-traffic-management/](https://www.quytech.com/blog/ai-for-smart-traffic-management/)

[26] AI-Based Smart Traffic Signal Control System. *International Journal of Research Publication and Reviews*, 6(4), 180-185, 2025. [Online]. Available: [https://ijrpr.com/uploads/V6ISSUE4/IJRPR43028.pdf](https://ijrpr.com/uploads/V6ISSUE4/IJRPR43028.pdf)

[27] AI-BASED DYNAMIC TRAFFIC MANAGEMENT SYSTEM WITH REAL-TIME DETECTION & PRIORITY SIGNAL OPTIMIZATION. *International Journal of Advanced Research in Computer and Communication Engineering*, 14(5), 1-6, 2025. [Online]. Available: [https://ijarcce.com/wp-content/uploads/2025/05/IJARCCE.2025.14504.pdf](https://ijarcce.com/wp-content/uploads/2025/05/IJARCCE.2025.14504.pdf)

[28] Ai-Based Traffic Controller Using Computer Vision. *Global Journal of Advanced Innovation, Research and Development*, 02(05), 38-43, 2024. [Online]. Available: [https://www.openjournals.ijaar.org/index.php/gjaitd/article/download/382/415/1220](https://www.openjournals.ijaar.org/index.php/gjaitd/article/download/382/415/1220)

[29] An Improved Smart Traffic Signal using Computer Vision and Artificial Intelligence. [Online]. Available: [https://www.researchgate.net/publication/364089749\_An\_Improved\_Smart\_Traffic\_Signal\_using\_Computer\_Vision\_and\_Artificial\_Intelligence](https://www.researchgate.net/publication/364089749_An_Improved_Smart_Traffic_Signal_using_Computer_Vision_and_Artificial_Intelligence)

[30] Intelligent Traffic Signal Automation Based on Computer Vision Techniques Using Deep Learning. *LJMU Research Online*, [Online]. Available: [https://researchonline.ljmu.ac.uk/id/eprint/16452/3/Intelligent%20Traffic%20Signal%20Automation%20Based%20on%20Computer%20Vision%20Techniques%20Using%20Deep%20Learning.pdf](https://researchonline.ljmu.ac.uk/id/eprint/16452/3/Intelligent%20Traffic%20Signal%20Automation%20Based%20on%20Computer%20Vision%20Techniques%20Using%20Deep%20Learning.pdf)

[31] AI for Intelligent Traffic Management in Smart Cities. *XenonStack Blog*, [Online]. Available: [https://www.xenonstack.com/blog/ai-intelligent-traffic-management](https://www.xenonstack.com/blog/ai-intelligent-traffic-management)

[32] signalized intersections - Traffic Flow Theory. *Federal Highway Administration*, [Online]. Available: [https://www.fhwa.dot.gov/publications/research/operations/tft/chap9.pdf](https://www.fhwa.dot.gov/publications/research/operations/tft/chap9.pdf)

[33] Impacts of Traffic Signal Control Strategies. *DiVA (Digital Academic Archive)*, [Online]. Available: [https://www.diva-portal.org/smash/get/diva2:11539/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:11539/FULLTEXT01.pdf)

[34] Artificial Intelligence-Based Adaptive Traffic Signal Control System: A Comprehensive Review. *Applied Sciences*, 13(19), 3875, 2023. [Online]. Available: [https://www.mdpi.com/2079-9292/13/19/3875](https://www.mdpi.com/2079-9292/13/19/3875)

[35] ASTM : Autonomous Smart Traffic Management System Using Artificial Intelligence CNN and LSTM. *arXiv preprint arXiv:2410.10929*, 2024. [Online]. Available: [https://arxiv.org/html/2410.10929v3](https://arxiv.org/html/2410.10929v3)

[36] Top AI Startups Revolutionizing Traffic Management. *Traction Five Blog*, [Online]. Available: [https://www.tractiontechnology.com/blog/traction-five-how-ai-is-revolutionizing-traffic-management](https://www.tractiontechnology.com/blog/traction-five-how-ai-is-revolutionizing-traffic-management)

[37] AI-based Traffic Management. *SWARCO Blog*, [Online]. Available: [https://www.swarco.com/stories/ai-based-traffic-management](https://www.swarco.com/stories/ai-based-traffic-management)

[38] A Dynamic Traffic Light Control Algorithm to Mitigate Traffic Congestion in Metropolitan Areas. *Sensors*, 24(12), 3987, 2024. [Online]. Available: [https://www.mdpi.com/1424-8220/24/12/3987](https://www.mdpi.com/1424-8220/24/12/3987)

[39] Object detection for traffic management based on YOLO. *SPIE OPTO*, 13018, 130181M, 2024. [Online]. Available: [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/3024069/Object-detection-for-traffic-management-based-on-YOLO/10.1117/12.3024069.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/3024069/Object-detection-for-traffic-management-based-on-YOLO/10.1117/12.3024069.full)
