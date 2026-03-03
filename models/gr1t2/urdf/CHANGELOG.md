# Important message
## Released Version
- **GR1T2_jaw**: version tab `10082024-alpha-1` has been released
- **GR1T2_fourier_hand**: version tab `10082024-alpha-2` has been released
- **GR1T2_inspire_hand**: version tab `10082024-alpha-1` has been released

## News
- **2024-09-27**
    - Version tab updated, all the version tab wil using `alpha` tab to verivy if it is a prerelease version and aims for test.


<!-- # GR1T2
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch. Version tab `09022024ver1`
- **2024-8-23**
    - Shuo Hu: Fiex the mass position error. Version tab `08232024ver1`.
- **2024-8-16**
    - Shuo Hu: updated the limit in `left(right)_upper_arm_roll`. Version tab `08162024ver1`.
- **2024-08-14**
    - Shuo Hu: Simplyfy the `left(right)_end_effector` description, updated the inertia in parallel link in foot and hand. Version tab `08142024ver1`
- **2024-08-06**
    - Shuo Hu: Fixed error in the mass center in urdf. Version tab `08062024ver1`
- **2024-08-05**
    - Shuo Hu: Swap the order between `left(right)_hand_roll` and `left(right)_hand_pitch` to align with the previous version URDF which has been released. Version tab `08052024ver1`
- **2024-08-01**
    - Shuo Hu: Fixed the error in frame at `right_lower_arm_link`. Version tab is `08012024ver1`.
    - Shuo Hu: Fixed the error in inertia, version tab `08012024ver2`
- **2024-07-31**
    - Shuo Hu: Updated GR1T2 with error frame direction at `right_shoulder_pitch_joint`. Added `left(right)_end_effector_link` at end of hand and connect to the `left(right)_hand_roll_link`. The end_effector aims to help people to add the hand or clap at the end of robot. Version tab is `07312024ver1`.
- **2024-07-29**
    - Shuo Hu: Uploaded GR1T2 URDF with version tab `07292024ver1`. It has been modified from the original URDF and rotor and stator has been seperated in the model. The model has no hand at end -->

<!-- 
# GR1T2_inspire_hand
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch. Version tab `09022024ver1`
- **2024-8-23**
    - Shuo Hu: Fiex the mass position error. Version tab `08232024ver1`.
- **2024-8-16**
    - Shuo Hu: updated the limit in `left(right)_upper_arm_roll`. Version tab `08162024ver1`.
- **2024-08-14**
    - Shuo Hu: Simplyfy the `left(right)_end_effector` description, updated the inertia in parallel link in foot and hand. Version tab`08142024ver1`
- **2024-08-06**
    - Shuo Hu: Fixed error in the mass center in urdf. Version tab `08062024ver1`
- **2024-08-05**
    - Shuo Hu: Swap the order between `left(right)_hand_roll` and `left(right)_hand_pitch` to align with the previous version URDF which has been released. Version tab `08052024ver1`
- **2024-08-01**
    - Shuo Hu: Fixed the error in frame at `right_lower_arm_link`. Version tab is `08012024ver1`.
    - Shuo Hu: Fixed the error in inertia, version tab `08012024ver2`
- **2024-07-31**
    - Shuo Hu: Updated GR1T2_hand with error frame direction at `right_shoulder_pitch_joint`. Added `left(right)_end_effector_link` at end of hand and connect to the `left(right)_hand_roll_link`. The end_effector aims to help people to add the hand or clap at the end of robot. Version tab is `07312024ver1`.
- **2024-07-29**
    - Shuo Hu: Uploaded GR1T2 URDF with version tab `07292024ver1`. It has been modified from the original URDF and rotor and stator has been seperated in the model. The model included dexterious hand(Inspire-robot) -->


# GR1T2
- **2025-01-25**
    - Shuo Hu: Refined the inertia at `foot_roll` and `foot_pitch`, the original inertia was treating them together as a whole model. Also done at `hand_roll` and `hand_pitch`. the new version seperated them just like other urdf. Version tab `01212025-alpha-1`
    - Shuo HU: Adding box collision define for GR1T2 `foot_roll` for checking collision with ground. The version tab is `01212025-alpha-2`
- **2024-10-08**
    - Shuo Hu: Redefined the joint limit at write_roll and wrist_pitch for more accurate result
- **2024-09-27**
    - Shuo Hu: Renamed the version tab from `09022024ver1` to `09022024-alpha-1` for alpha testing version of GR1T2 urdf
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch, and rename to `simple_coll` to align with next negeration. Version tab `09022024ver1`
- **2024-8-23**
    - Shuo Hu: Fiex the mass position error. Version tab `08232024ver1`.
- **2024-8-16**
    - Shuo Hu: updated the limit in `left(right)_upper_arm_roll`. Version tab `08162024ver1`.
- **2024-08-14**
    - Shuo Hu: Simplyfy the `left(right)_end_effector` description, updated the inertia in parallel link in foot and hand. Version tab`08142024ver1`
- **2024-08-06**
    - Shuo Hu: Fixed error in the mass center in urdf. Version tab `08062024ver1`
- **2024-08-05**
    - Shuo Hu: Creating the URDF with capsule collision checcking shape, this version has been made from `GR1T2`, so it has no hand. Version tab is `08052024ver1`.
    - Shuo Hu: Swap the order between `left(right)_hand_roll` and `left(right)_hand_pitch` to align with the previous version URDF which has been released. Version tab `08052024ver2`

# GR1T2_inspire_hand
**2025-01-25**
    - Shuo Hu: Refined the inertia at `foot_roll` and `foot_pitch`, the original inertia was treating them together as a whole model. Also done at `hand_roll` and `hand_pitch`. the new version seperated them just like other urdf. Version tab `01212025-alpha-1`
- **2024-10-08**
    - Shuo Hu: Redefined the joint limit at write_roll and wrist_pitch for more accurate result
- **2024-09-27**
    - Shuo Hu: Renamed the version tab from `09022024ver1` to `09022024-alpha-1` for alpha testing version of GR1T2 urdf
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch, and rename to `simple_coll` to align with next negeration. Version tab `09022024ver1`
- **2024-8-23**
    - Shuo Hu: Fiex the mass position error. Version tab `08232024ver1`.
- **2024-8-16**
    - Shuo Hu: updated the limit in `left(right)_upper_arm_roll`. Version tab `08162024ver1`.
- **2024-08-14**
    - Shuo Hu: Simplyfy the `left(right)_end_effector` description, updated the inertia in parallel link in foot and hand. Version tab`08142024ver1`
- **2024-08-06**
    - Shuo Hu: Fixed error in the mass center in urdf. Version tab `08062024ver1`
- **2024-08-05**
    - Shuo Hu: Creating the URDF with capsule collision checcking shape, this version has been made from `GR1T2_hand`, so it has inspired dexterous hand. Version tab is `08052024ver1`. Also, the order of between `left(right)_hand_roll` and `left(right)_hand_pitch` has been swaped

<!-- 
# GR1T2_jaw
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch. Version tab `09022024ver1`
- **2024-8-23**
    - Shuo Hu: Fiex the mass position error. Version tab `08232024ver1`.
- **2024-8-19**
    - Shuo Hu: Uploaded the urdf with jaw. Version tab `08192024ver1` -->

# GR1T2_jaw
**2025-01-25**
    - Shuo Hu: Refined the inertia at `foot_roll` and `foot_pitch`, the original inertia was treating them together as a whole model. Also done at `hand_roll` and `hand_pitch`. the new version seperated them just like other urdf. Version tab `01212025-alpha-1`
- **2024-10-08**
    - Shuo Hu: Redefined the joint limit at write_roll and wrist_pitch for more accurate result
- **2024-09-27**
    - Shuo Hu: Renamed the version tab from `09022024ver1` to `09022024-alpha-1` for alpha testing version of GR1T2 urdf
- **2024-09-02**
    - Shuo Hu: Simplified the `GR1T2_jaw` urdf collision part.

<!-- # GR1T2_fourier_hand
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch. Version tab `09022024ver1`
- **2024-08-24**
    -  Shuo Hu: Fixed the mesh error. Version tab `08242024ver1`
- **2024-8-23**
    - Shuo Hu: Fixed the mass position error. Version tab `08232024ver1`.
- **2024-08-20**
    - Shuo Hu: Uploaded the GR1T2 urdf with fourier hand, version tab `08202024ver1` -->

# GR1T2_fourier_hand_6dof
**2025-01-25**
    - Shuo Hu: Refined the inertia at `foot_roll` and `foot_pitch`, the original inertia was treating them together as a whole model. Also done at `hand_roll` and `hand_pitch`. the new version seperated them just like other urdf. Version tab `01212025-alpha-1`
- **2024-10-16**
    - Shuo Hu: GR1T2_fourier_hand has been updated, since the fourier hand `urdf` has updated, the version tab is `10162024-alpha-1`, name changed from `GR1T2_fourier_hand` to `GR1T2_fourier_hand_6dof`
- **2024-10-08**
    - Shuo Hu: Redefined the joint limit at write_roll and wrist_pitch for more accurate result
- **2024-10-08**
    - Shuo Hu: fourier_hand urdf has inertia error, has fixed. Version tab `10082024-alpha-1`
- **2024-009-29**
    - Shuo Hu: fourier_hand urdf has been updated, the bug at axis has been fixed, the version tab is `09292024-alpha-1`
- **2024-09-27**
    - Shuo Hu: Renamed the version tab from `09022024ver1` to `09022024-alpha-1` for alpha testing version of GR1T2 urdf
- **2024-09-02**
    - Shuo Hu: Fixed the mass error in thigh pitch, and rename to `simple_coll` to align with next negeration. Version tab `09022024ver1`
- **2024-08-24**
    -  Shuo Hu: Fixed the mesh error. Version tab `08242024ver1`
- **2024-8-23**
    - Shuo Hu: Updated urdf. Version tab `08232024ver1`.

# GR1T2_fourier_hand_12dof
**2025-01-25**
    - Shuo Hu: Refined the inertia at `foot_roll` and `foot_pitch`, the original inertia was treating them together as a whole model. Also done at `hand_roll` and `hand_pitch`. the new version seperated them just like other urdf. Version tab `01212025-alpha-1`
- **2024-10-16**
    - Shuo Hu: GR1T2_fourier_hand_12dof has been released, version tab is `10162024-alpha-1`
