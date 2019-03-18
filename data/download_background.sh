#!/bin/sh

echo "Downloading background.tar (part 1/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/k7t9hp7dpj71nqz4kfkq99ak4hbz3x6n.a
echo -n "Comparing checksum for background.tar (part 1/8)... "
if [ "$(md5sum k7t9hp7dpj71nqz4kfkq99ak4hbz3x6n.a | awk '{print $1}')" = ad06f4a3ff977067ba8072f23057e39a ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 2/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/po7trya3f4fmvv529ro85iumol05zsfc.b
echo -n "Comparing checksum for background.tar (part 2/8)... "
if [ "$(md5sum po7trya3f4fmvv529ro85iumol05zsfc.b | awk '{print $1}')" = 22c5bc1274f7adcef7fbe42363817e81 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 3/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/rm117qa7jc50vqub9ssbxg72r6g4hdpw.c
echo -n "Comparing checksum for background.tar (part 3/8)... "
if [ "$(md5sum rm117qa7jc50vqub9ssbxg72r6g4hdpw.c | awk '{print $1}')" = dd6b64235950fc7e04139e508a1ddd7c ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 4/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/stuewv6qliqoq72edw4gn4ms0yqs28hx.d
echo -n "Comparing checksum for background.tar (part 4/8)... "
if [ "$(md5sum stuewv6qliqoq72edw4gn4ms0yqs28hx.d | awk '{print $1}')" = ad06f4a3ff977067ba8072f23057e39a ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 5/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/yy2s6lbean9quadf7glsmf8h12ma6vd0.e
echo -n "Comparing checksum for background.tar (part 5/8)... "
if [ "$(md5sum yy2s6lbean9quadf7glsmf8h12ma6vd0.e | awk '{print $1}')" = 22c5bc1274f7adcef7fbe42363817e81 ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 6/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/rh7sxju3a641pbt1mhl8c4y25qhczyn5.f
echo -n "Comparing checksum for background.tar (part 6/8)... "
if [ "$(md5sum rh7sxju3a641pbt1mhl8c4y25qhczyn5.f | awk '{print $1}')" = dd6b64235950fc7e04139e508a1ddd7c ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 7/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/y6qyrsme1yfdx55kw6f9nzn8i3j5grqd.g
echo -n "Comparing checksum for background.tar (part 7/8)... "
if [ "$(md5sum y6qyrsme1yfdx55kw6f9nzn8i3j5grqd.g | awk '{print $1}')" = dd6b64235950fc7e04139e508a1ddd7c ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Downloading background.tar (part 8/8)..."
wget -q --show-progress https://cmu.box.com/shared/static/l0d642bkn70ho74uo4u7itnecgyr7uei.h
echo -n "Comparing checksum for background.tar (part 8/8)... "
if [ "$(md5sum l0d642bkn70ho74uo4u7itnecgyr7uei.h | awk '{print $1}')" = ad06f4a3ff977067ba8072f23057e39a ]; then echo "pass."; else echo "checksum doesn't match! Download may be corrupted."; fi

echo "Combining split files..."
cat k7t9hp7dpj71nqz4kfkq99ak4hbz3x6n.a \
	po7trya3f4fmvv529ro85iumol05zsfc.b \
	rm117qa7jc50vqub9ssbxg72r6g4hdpw.c \
	stuewv6qliqoq72edw4gn4ms0yqs28hx.d \
	yy2s6lbean9quadf7glsmf8h12ma6vd0.e \
	rh7sxju3a641pbt1mhl8c4y25qhczyn5.f \
	y6qyrsme1yfdx55kw6f9nzn8i3j5grqd.g \
	l0d642bkn70ho74uo4u7itnecgyr7uei.h > background.tar

echo "Removing split files..."
rm k7t9hp7dpj71nqz4kfkq99ak4hbz3x6n.a
rm po7trya3f4fmvv529ro85iumol05zsfc.b
rm rm117qa7jc50vqub9ssbxg72r6g4hdpw.c
rm stuewv6qliqoq72edw4gn4ms0yqs28hx.d
rm yy2s6lbean9quadf7glsmf8h12ma6vd0.e
rm rh7sxju3a641pbt1mhl8c4y25qhczyn5.f
rm y6qyrsme1yfdx55kw6f9nzn8i3j5grqd.g
rm l0d642bkn70ho74uo4u7itnecgyr7uei.h

echo "Done! Run \"tar -xf background.tar\" to extract files."
