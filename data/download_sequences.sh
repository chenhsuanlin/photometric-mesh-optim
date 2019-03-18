#!/bin/sh

echo "Downloading sequences.tar.gz (part 1/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/6wm2scui2ge7i1486j6zwhbm9cbg4wfg.a
echo -n "Comparing checksum for sequences.tar.gz (part 1/6)... "
if [ "$(md5sum 6wm2scui2ge7i1486j6zwhbm9cbg4wfg.a | awk '{print $1}')" = 1afaf12f032dd21f4a1fae649aa2446b ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading sequences.tar.gz (part 2/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/pr1ys5qq9wgmtvwpcfqjzd0x13qr6qtj.b
echo -n "Comparing checksum for sequences.tar.gz (part 2/6)... "
if [ "$(md5sum pr1ys5qq9wgmtvwpcfqjzd0x13qr6qtj.b | awk '{print $1}')" = 9b7ee24dd778a4b2992b997f985e1b5a ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading sequences.tar.gz (part 3/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/7w3263lmtf0u27dp37ja089cc69cr57s.c
echo -n "Comparing checksum for sequences.tar.gz (part 3/6)... "
if [ "$(md5sum 7w3263lmtf0u27dp37ja089cc69cr57s.c | awk '{print $1}')" = 8d0d5e116284c2949d83d438e7e2d7e9 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading sequences.tar.gz (part 4/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/8is0z63ijkkmygtozijtej79hd0rzglt.d
echo -n "Comparing checksum for sequences.tar.gz (part 4/6)... "
if [ "$(md5sum 8is0z63ijkkmygtozijtej79hd0rzglt.d | awk '{print $1}')" = 4e79b8d28c78d1d1b5108327f7b47631 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading sequences.tar.gz (part 5/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/apa9w8ehh3p21ava83ec8rm4ya4iok9g.e
echo -n "Comparing checksum for sequences.tar.gz (part 5/6)... "
if [ "$(md5sum apa9w8ehh3p21ava83ec8rm4ya4iok9g.e | awk '{print $1}')" = 06895e7c7b247e54b7a91f0c0bcbcd68 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading sequences.tar.gz (part 6/6)..."
wget -q --show-progress https://cmu.box.com/shared/static/3d9badm0v222ud1h3jj4r3r2jcvrj7hx.f
echo -n "Comparing checksum for sequences.tar.gz (part 6/6)... "
if [ "$(md5sum 3d9badm0v222ud1h3jj4r3r2jcvrj7hx.f | awk '{print $1}')" = 4aae66e91e731d9c22d58e605c524d43 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Combining split files..."
cat 6wm2scui2ge7i1486j6zwhbm9cbg4wfg.a \
	pr1ys5qq9wgmtvwpcfqjzd0x13qr6qtj.b \
	7w3263lmtf0u27dp37ja089cc69cr57s.c \
	8is0z63ijkkmygtozijtej79hd0rzglt.d \
	apa9w8ehh3p21ava83ec8rm4ya4iok9g.e \
	3d9badm0v222ud1h3jj4r3r2jcvrj7hx.f > sequences.tar.gz

echo "Removing split files..."
rm 6wm2scui2ge7i1486j6zwhbm9cbg4wfg.a
rm pr1ys5qq9wgmtvwpcfqjzd0x13qr6qtj.b
rm 7w3263lmtf0u27dp37ja089cc69cr57s.c
rm 8is0z63ijkkmygtozijtej79hd0rzglt.d
rm apa9w8ehh3p21ava83ec8rm4ya4iok9g.e
rm 3d9badm0v222ud1h3jj4r3r2jcvrj7hx.f

echo "Done! Run \"tar -zxf sequences.tar.gz\" to extract files."
