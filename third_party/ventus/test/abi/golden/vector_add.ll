; ModuleID = 'third_party/ventus/test/abi/vector_add.cl'
source_filename = "third_party/ventus/test/abi/vector_add.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"
target triple = "riscv32"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) vscale_range(1,2048)
define dso_local ventus_kernel void @vector_add(ptr addrspace(1) nocapture noundef readonly align 4 %0, ptr addrspace(1) nocapture noundef readonly align 4 %1, ptr addrspace(1) nocapture noundef writeonly align 4 %2, i32 noundef %3) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %5 = call i32 @_Z13get_global_idj(i32 noundef 0) #2
  %6 = icmp slt i32 %5, %3
  br i1 %6, label %7, label %14

7:                                                ; preds = %4
  %8 = getelementptr inbounds float, ptr addrspace(1) %2, i32 %5
  %9 = getelementptr inbounds float, ptr addrspace(1) %1, i32 %5
  %10 = getelementptr inbounds float, ptr addrspace(1) %0, i32 %5
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !10
  %12 = load float, ptr addrspace(1) %9, align 4, !tbaa !10
  %13 = fadd float %11, %12
  store float %13, ptr addrspace(1) %8, align 4, !tbaa !10
  br label %14

14:                                               ; preds = %7, %4
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i32 @_Z13get_global_idj(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) vscale_range(1,2048) "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" }
attributes #2 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, !"SmallDataLimit", i32 8}
!4 = !{i32 2, i32 0}
!5 = !{!"clang version 16.0.0 (https://github.com/THU-DSP-LAB/llvm-project.git d4f2063fe81cbbefda34da60d4cd5c46bce3d231)"}
!6 = !{i32 1, i32 1, i32 1, i32 0}
!7 = !{!"none", !"none", !"none", !"none"}
!8 = !{!"float*", !"float*", !"float*", !"int"}
!9 = !{!"const", !"const", !"", !""}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
