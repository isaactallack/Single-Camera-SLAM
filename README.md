# Simultaneous Localization and Mapping (SLAM) in a single camera system

Human activity recognition (HAR) is a tool that has become increasingly common and easier
to perform over the last few years with the huge growth of people using smartphones equipped
with a wide range of different sensors and access to the internet. The uses for HAR are countless
including healthcare, fitness and advertising. However, HAR algorithms are not always perfect
and smartphone sensors produce a huge amount of data every second meaning there is a lot
of interest around improving these classification algorithms not only for increased accuracy but
also for speed and efficiency, particularly important for lower-power devices with small batteries
like smartphones.

The aim of this report was to test and evaluate machine learning and data science methods
to classify the activity in a given time window from sensor data generated from a smartphone
(particularly accelerometer and gyroscope data). Different approaches to this problem are tested
with the aim of maximising classification accuracy while keeping speed and efficiency in mind.
The algorithms must be robust and generic enough classify unseen or unusual data as this will
be vital in a real-world implementation.

The analysis found that very high performing classification algorithms could be created to not
only classify activities but extended to detect falls with extremely high accuracy as well. A
wide range of algorithms were tested but overall it seemed that the most information could be
inferred from the frequency-domain representation of the accelerometer signals as these were
the algorithms that performed best throughout. The main achievements of this report involved
producing a fall detection classifier which performed 100% precision with minimal amounts of
data in training/testing as well as an activity classifier that performed 99% precision classifying
between five different activities.
