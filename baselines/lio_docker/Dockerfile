# #083: LiDAR-inertial references (COIN-LIO, FAST-LIO2) in one ROS Noetic workspace, from local clones (pinned commits).
FROM ros:noetic-perception
RUN apt-get -qq update && apt-get install -y -q python3-catkin-tools libgoogle-glog-dev git cmake build-essential \
    ros-noetic-tf ros-noetic-eigen-conversions ros-noetic-pcl-ros && rm -rf /var/lib/apt/lists/*
# Livox SDK + livox_ros_driver: FAST-LIO2 needs its CustomMsg even for Ouster / Hesai input
RUN git clone -q --depth 1 https://github.com/Livox-SDK/Livox-SDK.git /root/Livox-SDK && cd /root/Livox-SDK/build && \
    cmake .. -DCMAKE_CXX_FLAGS="-include cstdint" > /dev/null && make -j8 > /dev/null && make install > /dev/null
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src && cd $CATKIN_WS/src && git clone -q --depth 1 https://github.com/Livox-SDK/livox_ros_driver.git
COPY COIN-LIO $CATKIN_WS/src/COIN-LIO
COPY FAST_LIO $CATKIN_WS/src/FAST_LIO
RUN cd $CATKIN_WS && catkin init > /dev/null && catkin config --extend /opt/ros/noetic --merge-devel \
    --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null && (catkin build -j8 --no-status livox_ros_driver coin_lio fast_lio || catkin build -j8 --no-status livox_ros_driver coin_lio fast_lio)  # 2nd pass: COIN-LIO msg headers race
RUN echo 'source /root/catkin_ws/devel/setup.bash' >> /root/.bashrc
