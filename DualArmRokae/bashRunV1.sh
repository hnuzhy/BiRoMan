
# pass arguments to the script
#!/bin/sh
# --task_name --no_interaction --test_id --object_id1 --object_id2
echo $1 $2 $3 $4 $5

if [ "$2" == "1" ]; then  
    interaction_cfg=""
elif [ "$2" == "0" ]; then 
    interaction_cfg="--no_interaction"
else  
    echo "Unknown Argument!!!"
fi 

######################################################################################################################################################
if [ "$1" == "pouring" ]; then          # ./bashRunV1.sh pouring 1 10 1 1
    python skills/runScriptV1.py --task_name pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5
elif [ "$1" == "unscrew" ]; then        # ./bashRunV1.sh unscrew 1 10 1
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "reorient" ]; then       # ./bashRunV1.sh reorient 1 10 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "grasping" ]; then       # ./bashRunV1.sh grasping 1 10 1
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $4
elif [ "$1" == "flatting" ]; then       # ./bashRunV1.sh flatting 1 10 1
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "flipping" ]; then       # ./bashRunV1.sh flipping 1 10 1
    python skills/runScriptV1.py --task_name flipping $interaction_cfg --test_id $3 --mugcup_id $4

elif [ "$1" == "grasping_rectbox" ]; then       # ./bashRunV1.sh grasping_rectbox 1 10 1
    python skills/runScriptV1.py --task_name grasping_rectbox $interaction_cfg --test_id $3 --rectbox_id $4
elif [ "$1" == "grasping_cirbowl" ]; then       # ./bashRunV1.sh grasping_cirbowl 1 10 1
    python skills/runScriptV1.py --task_name grasping_cirbowl $interaction_cfg --test_id $3 --cirbowl_id $4
elif [ "$1" == "grasping_basket" ]; then        # ./bashRunV1.sh grasping_basket 1 10 1
    python skills/runScriptV1.py --task_name grasping_basket $interaction_cfg --test_id $3 --basket_id $4
elif [ "$1" == "grasping_holder" ]; then        # ./bashRunV1.sh grasping_holder 1 10 1
    python skills/runScriptV1.py --task_name grasping_holder $interaction_cfg --test_id $3 --holder_id $4
elif [ "$1" == "grasping_pencup" ]; then        # ./bashRunV1.sh grasping_pencup 1 10 1
    python skills/runScriptV1.py --task_name grasping_pencup $interaction_cfg --test_id $3 --pencup_id $4

elif [ "$1" == "unscrew-pouring" ]; then        # ./bashRunV1.sh unscrew-pouring 1 10 1 1
    python skills/runScriptV1.py --task_name unscrew-pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5
elif [ "$1" == "flatting-reorient" ]; then      # ./bashRunV1.sh flatting-reorient 1 10 1
    python skills/runScriptV1.py --task_name flatting-reorient $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "flipping-grasping" ]; then      # ./bashRunV1.sh flipping-grasping 1 10 1
    python skills/runScriptV1.py --task_name flipping-grasping $interaction_cfg --test_id $3 --mugcup_id $4

elif [ "$1" == "inserting" ]; then          # ./bashRunV1.sh inserting 1 10 1 1
    python skills/runScriptV1.py --task_name inserting $interaction_cfg --test_id $3 --marker_id $4 --ordcup_id $5
elif [ "$1" == "plugpen" ]; then          # ./bashRunV1.sh plugpen 1 10 1 1
    python skills/runScriptV1.py --task_name plugpen $interaction_cfg --test_id $3 --pencap_id $4 --marker_id $5
elif [ "$1" == "handover" ]; then          # ./bashRunV1.sh handover 1 10 1
    python skills/runScriptV1.py --task_name handover $interaction_cfg --test_id $3 --shovel_id $4
	
elif [ "$1" == "ppspoon" ]; then            # ./bashRunV1.sh ppspoon 1 10 1
    python skills/runScriptV1.py --task_name ppspoon $interaction_cfg --test_id $3 --spoon_id $4
elif [ "$1" == "ppfork" ]; then             # ./bashRunV1.sh ppfork 1 10 1
    python skills/runScriptV1.py --task_name ppfork $interaction_cfg --test_id $3 --fork_id $4
elif [ "$1" == "ppspoon-ppfork" ]; then     # ./bashRunV1.sh ppspoon-ppfork 1 10 1 1
    python skills/runScriptV1.py --task_name ppspoon-ppfork $interaction_cfg --test_id $3 --spoon_id $4 --fork_id $5
elif [ "$1" == "ppfork-ppspoon" ]; then     # ./bashRunV1.sh ppfork-ppspoon 1 10 1 1
    python skills/runScriptV1.py --task_name ppfork-ppspoon $interaction_cfg --test_id $3 --fork_id $4 --spoon_id $5

elif [ "$1" == "ppspoon_ppfork" ]; then     # ./bashRunV1.sh ppspoon_ppfork 1 10 1 1
    python skills/runScriptV1.py --task_name ppspoon $interaction_cfg --test_id $3 --spoon_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name ppfork $interaction_cfg --test_id $3 --fork_id $5 --is_syn

elif [ "$1" == "pivoting" ]; then           # ./bashRunV1.sh pivoting 1 10 1
    python skills/runScriptV1.py --task_name pivoting $interaction_cfg --test_id $3 --cirbowl_id $4
elif [ "$1" == "wrapping" ]; then           # ./bashRunV1.sh wrapping 1 10 1
    python skills/runScriptV1.py --task_name wrapping $interaction_cfg --test_id $3 --basket_id $4

elif [ "$1" == "pivoting_rectbox" ]; then       # ./bashRunV1.sh pivoting_rectbox 1 10 1
    python skills/runScriptV1.py --task_name pivoting_rectbox $interaction_cfg --test_id $3 --rectbox_id $4
elif [ "$1" == "pivoting_cirbowl" ]; then       # ./bashRunV1.sh pivoting_cirbowl 1 10 1
    python skills/runScriptV1.py --task_name pivoting_cirbowl $interaction_cfg --test_id $3 --cirbowl_id $4
elif [ "$1" == "flipping_basket" ]; then        # ./bashRunV1.sh flipping_basket 1 10 1
    python skills/runScriptV1.py --task_name flipping_basket $interaction_cfg --test_id $3 --basket_id $4
elif [ "$1" == "flipping_block" ]; then        # ./bashRunV1.sh flipping_block 1 10 1
    python skills/runScriptV1.py --task_name flipping_block $interaction_cfg --test_id $3 --block_id $4
elif [ "$1" == "pivoting_bigjar" ]; then        # ./bashRunV1.sh pivoting_bigjar 1 10 1
    python skills/runScriptV1.py --task_name pivoting_bigjar $interaction_cfg --test_id $3 --bigjar_id $4
elif [ "$1" == "pivoting_block" ]; then        # ./bashRunV1.sh pivoting_block 1 10 1
    python skills/runScriptV1.py --task_name pivoting_block $interaction_cfg --test_id $3 --block_id $4
elif [ "$1" == "toppling_holder" ]; then        # ./bashRunV1.sh toppling_holder 1 10 1
    python skills/runScriptV1.py --task_name toppling_holder $interaction_cfg --test_id $3 --holder_id $4
elif [ "$1" == "toppling_bigjar" ]; then        # ./bashRunV1.sh toppling_bigjar 1 10 1
    python skills/runScriptV1.py --task_name toppling_bigjar $interaction_cfg --test_id $3 --bigjar_id $4
elif [ "$1" == "bilifting_bigjar" ]; then        # ./bashRunV1.sh bilifting_bigjar 1 10 1
    python skills/runScriptV1.py --task_name bilifting_bigjar $interaction_cfg --test_id $3 --bigjar_id $4
elif [ "$1" == "bilifting_block" ]; then        # ./bashRunV1.sh bilifting_block 1 10 1
    python skills/runScriptV1.py --task_name bilifting_block $interaction_cfg --test_id $3 --block_id $4

###########################################################################

elif [ "$1" == "penbagzip" ]; then          # ./bashRunV1.sh penbagzip 1 10 1 1
    python skills/runScriptV1.py --task_name penbagzip $interaction_cfg --test_id $3 --penbag_id $4 --marker_id $5
elif [ "$1" == "foldtowel" ]; then          # ./bashRunV1.sh foldtowel 1 10 1
    python skills/runScriptV1.py --task_name foldtowel $interaction_cfg --test_id $3 --towel_id $4
elif [ "$1" == "foldpants" ]; then          # ./bashRunV1.sh foldpants 1 10 1
    python skills/runScriptV1.py --task_name foldpants $interaction_cfg --test_id $3 --pants_id $4
elif [ "$1" == "foldshirt" ]; then          # ./bashRunV1.sh foldshirt 1 10 1
    python skills/runScriptV1.py --task_name foldshirt $interaction_cfg --test_id $3 --shirt_id $4
elif [ "$1" == "coilcable" ]; then          # ./bashRunV1.sh coilcable 1 10 1
    python skills/runScriptV1.py --task_name coilcable $interaction_cfg --test_id $3 --cable_id $4
elif [ "$1" == "coilrope" ]; then          # ./bashRunV1.sh coilrope 1 10 1
    python skills/runScriptV1.py --task_name coilrope $interaction_cfg --test_id $3 --rope_id $4
elif [ "$1" == "coilbelt" ]; then          # ./bashRunV1.sh coilbelt 1 10 1
    python skills/runScriptV1.py --task_name coilbelt $interaction_cfg --test_id $3 --belt_id $4
	
###########################################################################

elif [ "$1" == "flatting_flipping" ]; then                  # ./bashRunV1.sh flatting_flipping 1 100 1 1
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name flipping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
elif [ "$1" == "flatting_reorient" ]; then                  # ./bashRunV1.sh flatting_reorient 1 100 1
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "flipping_grasping" ]; then                   # ./bashRunV1.sh flipping_grasping 1 100 1
    python skills/runScriptV1.py --task_name flipping $interaction_cfg --test_id $3 --mugcup_id $4
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $4
elif [ "$1" == "reorient_grasping" ]; then                  # ./bashRunV1.sh reorient_grasping 1 100 1 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
elif [ "$1" == "reorient_unscrew" ]; then                   # ./bashRunV1.sh reorient_unscrew 1 100 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "grasping_pouring" ]; then                   # ./bashRunV1.sh grasping_pouring 1 100 1 1
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $5
    python skills/runScriptV1.py --task_name pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5
elif [ "$1" == "unscrew_pouring" ]; then                    # ./bashRunV1.sh unscrew_pouring 1 100 1 1
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################

elif [ "$1" == "reorient_unscrew_pouring" ]; then           # ./bashRunV1.sh reorient_unscrew_pouring 1 100 1 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5
elif [ "$1" == "flatting_reorient_unscrew" ]; then          # ./bashRunV1.sh flatting_reorient_unscrew 1 100 1
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
elif [ "$1" == "reorient-unscrew-pouring" ]; then       # ./bashRunV1.sh reorient-unscrew-pouring 1 10 1 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4 
    python skills/runScriptV1.py --task_name unscrew-pouring $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################

elif [ "$1" == "flatting_flipping_reorient_grasping" ]; then    # ./bashRunV1.sh flatting_flipping_reorient_grasping 1 100 1 1
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name flipping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
elif [ "$1" == "reorient_grasping_unscrew_pouring" ]; then      # ./bashRunV1.sh reorient_grasping_unscrew_pouring 1 100 1 1
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name pouring --no_interaction --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################

# ./bashRunV1.sh flatting_flipping_reorient_grasping_unscrew_pouring 1 100 1 1
elif [ "$1" == "flatting_flipping_reorient_grasping_unscrew_pouring" ]; then  
    python skills/runScriptV1.py --task_name flatting $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name flipping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
    python skills/runScriptV1.py --task_name reorient $interaction_cfg --test_id $3 --bottle_id $4 --is_syn & \
    python skills/runScriptV1.py --task_name grasping $interaction_cfg --test_id $3 --mugcup_id $5 --is_syn
    python skills/runScriptV1.py --task_name unscrew $interaction_cfg --test_id $3 --bottle_id $4
    python skills/runScriptV1.py --task_name pouring --no_interaction --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################
###########################################################################

elif [ "$1" == "pouring_full_stage" ]; then  # ./bashRunV1.sh pouring_full_stage 1 1000 1 1
    python skills/pourClosedLoopV1.py --task_name pouring_full_stage $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################
###########################################################################

elif [ "$1" == "urm_t1_box" ]; then             # ./bashRunV1.sh urm_t1_box 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t1_box $interaction_cfg --test_id $3 --rectbox_id $4
elif [ "$1" == "urm_t2_bowl" ]; then            # ./bashRunV1.sh urm_t2_bowl 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t2_bowl $interaction_cfg --test_id $3 --cirbowl_id $4
elif [ "$1" == "urm_t3_basket" ]; then          # ./bashRunV1.sh urm_t3_basket 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t3_basket $interaction_cfg --test_id $3 --basket_id $4
elif [ "$1" == "urm_t4_holder" ]; then          # ./bashRunV1.sh urm_t4_holder 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t4_holder $interaction_cfg --test_id $3 --holder_id $4
elif [ "$1" == "urm_t5_bigjar" ]; then          # ./bashRunV1.sh urm_t5_bigjar 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t5_bigjar $interaction_cfg --test_id $3 --bigjar_id $4
elif [ "$1" == "urm_t6_block" ]; then           # ./bashRunV1.sh urm_t6_block 1 1000 1
    python skills/urmLongTasks.py --task_name urm_t6_block $interaction_cfg --test_id $3 --block_id $4

###########################################################################

elif [ "$1" == "rarg_t1_dining" ]; then		# ./bashRunV1.sh rarg_t1_dining 1 1000 1 1 1
    python skills/rearrangement.py --task_name rarg_t1_dining $interaction_cfg --test_id $3 --cirbowl_id $4 --spoon_id $5 --fork_id $6
elif [ "$1" == "rarg_t2_drinking" ]; then	# ./bashRunV1.sh rarg_t2_drinking 1 1000 1 1
    python -+/rearrangement.py --task_name rarg_t2_drinking $interaction_cfg --test_id $3 --bottle_id $4 --mugcup_id $5

###########################################################################

else  
    echo "Unknown Argument for --task_name!!!"    
fi 
######################################################################################################################################################

# ./bashRunV1.sh reorient 1 20 1
# ./bashRunV1.sh wrapping 1 20 3
# ./bashRunV1.sh unscrew-pouring 1 20 1 1
# ./bashRunV1.sh inserting 1 20 1 1


