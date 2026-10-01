#!/usr/bin/env bash
# #083: one LIO run inside kiss-lio:1.  run_in_container.sh <coin|fast> <out dir> <launch args...> -- <bag files...>
# Starts the method's mapping node with its own config, plays the bag(s) at real time with --clock, records /Odometry.
source /root/catkin_ws/devel/setup.bash
method=$1; out=$2; shift 2
args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift; bags=("$@")
mkdir -p "$out"
roscore > "$out/roscore.log" 2>&1 & sleep 4
rosparam set use_sim_time true
if [ "$method" = coin ]; then roslaunch coin_lio "${args[@]}" rviz:=false > "$out/method.log" 2>&1 &
else roslaunch "${args[@]}" > "$out/method.log" 2>&1 & fi
sleep 8
rosbag record -O "$out/odometry.bag" /Odometry > "$out/record.log" 2>&1 &
sleep 2
rosbag play --clock --quiet --queue=1000 "${bags[@]}" > "$out/play.log" 2>&1
echo "play exit $?" >> "$out/play.log"
sleep 10; pkill -INT -f "rosbag record"; sleep 5; pkill -INT -f roslaunch; sleep 5; pkill -f rosmaster
