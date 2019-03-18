#!/bin/sh

echo "Downloading rendering.tar (part 1/3)..."
wget -q --show-progress https://cmu.box.com/shared/static/z2qw9hpnsi7e7h58d5b6l6xeherjyl3f.a
echo -n "Comparing checksum for rendering.tar (part 1/3)... "
if [ "$(md5sum z2qw9hpnsi7e7h58d5b6l6xeherjyl3f.a | awk '{print $1}')" = ad06f4a3ff977067ba8072f23057e39a ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading rendering.tar (part 2/3)..."
wget -q --show-progress https://cmu.box.com/shared/static/4elvxp75o90x8frgpz9qzz0lz6v8wdrh.b
echo -n "Comparing checksum for rendering.tar (part 2/3)... "
if [ "$(md5sum 4elvxp75o90x8frgpz9qzz0lz6v8wdrh.b | awk '{print $1}')" = 22c5bc1274f7adcef7fbe42363817e81 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading rendering.tar (part 3/3)..."
wget -q --show-progress https://cmu.box.com/shared/static/6ejw9k0as8800mrnmn03f4jio4uajotw.c
echo -n "Comparing checksum for rendering.tar (part 3/3)... "
if [ "$(md5sum 6ejw9k0as8800mrnmn03f4jio4uajotw.c | awk '{print $1}')" = dd6b64235950fc7e04139e508a1ddd7c ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Combining split files..."
cat z2qw9hpnsi7e7h58d5b6l6xeherjyl3f.a \
	4elvxp75o90x8frgpz9qzz0lz6v8wdrh.b \
	6ejw9k0as8800mrnmn03f4jio4uajotw.c > rendering.tar

echo "Removing split files..."
rm z2qw9hpnsi7e7h58d5b6l6xeherjyl3f.a
rm 4elvxp75o90x8frgpz9qzz0lz6v8wdrh.b
rm 6ejw9k0as8800mrnmn03f4jio4uajotw.c

echo "Done! Run \"tar -xf rendering.tar\" to extract files."
