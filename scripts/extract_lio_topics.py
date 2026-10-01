#!/usr/bin/env python3
"""Copy the LiDAR and IMU topics of a (possibly bz2-compressed, split) ROS1 sequence into one uncompressed bag (#083).

    python scripts/extract_lio_topics.py <bag or folder of bags> <out.bag> <topic> [<topic> ...]

The Oxford Spires and Hilti 2021 bags are bz2-compressed with all camera images in the same chunks: `rosbag play` cannot
decompress them at real time once a LIO node subscribes, so the player stalls (FAST-LIO2 got no scan on christ-church-03).
The raw messages are copied byte for byte (no deserialisation), with their record times and connection headers, so the
LIO methods receive exactly the original messages.  Output on the external SSD, read-only input.
"""
import sys
from pathlib import Path

from rosbags.rosbag1 import Reader, Writer


def main():
    src, out, topics = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
    bags = sorted(src.glob("*.bag")) if src.is_dir() else [src]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    if tmp.exists():
        tmp.unlink()
    n = {}
    with Writer(tmp) as writer:
        conns = {}
        for bag in bags:
            with Reader(bag) as reader:
                for c in [c for c in reader.connections if c.topic in topics]:
                    key = (c.topic, c.msgtype)
                    if key not in conns:
                        conns[key] = writer.add_connection(c.topic, c.msgtype, msgdef=getattr(c.msgdef, "data", c.msgdef), md5sum=c.digest,
                                                           callerid=c.ext.callerid, latching=c.ext.latching)
                for c, t, raw in reader.messages(connections=[c for c in reader.connections if c.topic in topics]):
                    writer.write(conns[(c.topic, c.msgtype)], t, raw)
                    n[c.topic] = n.get(c.topic, 0) + 1
    tmp.rename(out)
    print(f"{out}: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))


if __name__ == "__main__":
    main()
